#ifndef KERNIX_MODEL_H
#define KERNIX_MODEL_H

/* 모델 추론 런타임: .knx 파일(tools/convert.py로 PyTorch·Keras 모델에서 내보냄)을 읽어 실행.
 * 설계: docs/mlp-runtime-design.md (KNX1), docs/int8-quantization-design.md (KNX2), docs/graph-runtime-design.md (KNX3).
 *
 * 값(텐서)은 표본마다 channels-last 모양(예: H×W×C, T×F)을 갖고, 버퍼에는 (배치·앞 축들의 곱) × 채널의 2차원으로 둔다.
 * 그래서 합성곱은 im2col 뒤 행렬곱 하나(GPU+AMX), Flatten/Reshape는 복사 없는 보기 변경이다.
 * 연산은 입력 값 번호를 갖는 그래프(KNX3)라 잔차 연결·어텐션도 표현한다. 로더가 값의 수명을 보고 저장 버퍼를 재사용하고,
 * 가능한 연산은 제자리로 처리하며, 행렬곱/합성곱 바로 뒤 활성화는 합친다. 배치 크기가 그대로면 forward 중 할당이 없고,
 * 큰 배치는 작업 버퍼 상한(ws_limit) 안에서 나눠 실행한다. */

#include "kernix.h"
#include <limits.h>
#include <float.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if !defined(KERNIX_ACCELERATE_ENABLED)
#error "kernix_model.h는 -DKERNIX_USE_ACCELERATE가 필요합니다"
#endif

/* 파일에 쓰이는 번호: 바꾸면 기존 .knx와 호환되지 않음 */
enum {
    KERNIX_OP_LINEAR, KERNIX_OP_ACT, KERNIX_OP_LAYERNORM, KERNIX_OP_SOFTMAX, KERNIX_OP_LSTM, /* KNX1부터 */
    KERNIX_OP_GRU, KERNIX_OP_RNN, KERNIX_OP_CONV, KERNIX_OP_POOL, KERNIX_OP_GLOBAL_POOL,     /* 이하 KNX3 */
    KERNIX_OP_RESHAPE, KERNIX_OP_PERMUTE, KERNIX_OP_BINARY, KERNIX_OP_CONCAT, KERNIX_OP_EMBEDDING,
    KERNIX_OP_AFFINE, KERNIX_OP_ADD_CONST, KERNIX_OP_ATTENTION, KERNIX_OP_SELECT, KERNIX_OP_PAD,
    KERNIX_OP_UPSAMPLE,
    KERNIX_OP_LAST = KERNIX_OP_UPSAMPLE
};
/* 활성화 번호 KERNIX_ACT_*는 kernix.h */
enum { KERNIX_BIN_ADD, KERNIX_BIN_SUB, KERNIX_BIN_MUL, KERNIX_BIN_DIV, KERNIX_BIN_MAX, KERNIX_BIN_MIN };
enum { KERNIX_MERGE_CONCAT, KERNIX_MERGE_SUM, KERNIX_MERGE_MUL, KERNIX_MERGE_AVE }; /* 양방향 RNN 출력 합치기 */
enum { KERNIX_POOL_MAX, KERNIX_POOL_AVG };
enum { KERNIX_AXIS_FIRST, KERNIX_AXIS_LAST }; /* CONCAT/SELECT: 표본의 첫 축 또는 마지막(채널) 축 */
#define KERNIX_MAX_DIMS 4   /* 표본 하나의 축 수 상한 (배치 제외) */
#define KERNIX_MAX_INPUTS 8 /* 연산 하나의 입력 수 상한 (CONCAT) */
#define KERNIX_MAX_GEOM 65536 /* 합성곱/풀링 커널·보폭·패딩 상한 */

#define KERNIX_MODEL_MAX_OPS 1024

/* ---- int8 가중치 (docs/int8-quantization-design.md) ----
 * 가중치는 출력 채널별 scale을 둔 대칭 int8, 입력은 행마다 동적 int8(sx = max|x|/127)로 바꿔 int32로 누적한다.
 * 행이 적은 곱(웹캠 추론의 LSTM 순환, Dense)에서 AMX float보다 빠르다. 행이 많으면 AMX가 빠르다 */
#define KERNIX_INT8_MAX_K 131072 /* 128·127·K < 2³¹: int32 누적이 넘치지 않는 K 상한 */
/* 이 행 수 이하만 int8 커널. M1 측정(ns int8/float): 64×256 1행 194/611, 2행 373/702, 4행 732/472;
 * 128×512 1행 641/4165, 2행 1283/1844, 4행 2545/1894; 784×1024 1행 10512/18345, 2행 20955/31818, 4행 41875/32167 */
#define KERNIX_INT8_MAX_ROWS 2

typedef struct kernix_qweight {
    int k, n;      /* 실제 크기 (입력 × 출력) */
    int kp, np;    /* 16의 배수로 늘린 크기, 늘린 칸은 0 */
    int8_t* q;     /* sdot 묶음: 출력 16개 블록 → k 4개 블록 → 벡터 4개(출력 4개 × k 4개) */
    float* scale;  /* np개, 늘린 칸은 0 */
} kernix_qweight;

static inline void kernix_qweight_free(kernix_qweight* w)
{
    if (w == NULL) return;
    free(w->q);
    free(w->scale);
    free(w);
}

static inline size_t kernix_qweight_at(const kernix_qweight* w, int k, int n)
{
    return (((size_t)(n / 16) * (size_t)(w->kp / 4) + (size_t)(k / 4)) * 4 + (size_t)((n % 16) / 4)) * 16 + (size_t)(n % 4) * 4 + (size_t)(k % 4);
}

/* q: k×n 행 우선(in×out), scale: n개 (w[k][n] ≈ q[k][n]·scale[n]). 실패 시 NULL */
static inline kernix_qweight* kernix_qweight_pack(const int8_t* q, const float* scale, int k, int n)
{
    if (q == NULL || scale == NULL || k < 1 || n < 1 || k > KERNIX_INT8_MAX_K || n > INT_MAX - 16) return NULL;
    kernix_qweight* w = calloc(1, sizeof *w);
    if (w == NULL) return NULL;
    w->k = k; w->n = n;
    w->kp = (k + 15) / 16 * 16;
    w->np = (n + 15) / 16 * 16;
    w->q = calloc((size_t)w->kp * (size_t)w->np, 1);
    w->scale = calloc((size_t)w->np, sizeof(float));
    if (w->q == NULL || w->scale == NULL) { kernix_qweight_free(w); return NULL; }
    memcpy(w->scale, scale, sizeof(float) * (size_t)n);
    for (int i = 0; i < k; i++)
        for (int j = 0; j < n; j++) w->q[kernix_qweight_at(w, i, j)] = q[(size_t)i * n + j];
    return w;
}

/* dst: k×n float (q·scale) */
static inline void kernix_qweight_unpack(const kernix_qweight* w, float* dst)
{
    for (int i = 0; i < w->k; i++)
        for (int j = 0; j < w->n; j++) dst[(size_t)i * w->n + j] = (float)w->q[kernix_qweight_at(w, i, j)] * w->scale[j];
}

/* y[i] = (accumulate ? y[i] : 0) + x[i]·W + (bias ? bias : 0), i < rows. x: rows×k, y: rows×n.
 * NaN/무한대가 있는 입력 행은 출력 행 전체가 NaN (반올림 변환이 NaN을 0으로 바꾸므로 따로 처리). 성공 0, 메모리 부족 -1 */
static inline int kernix_qmatmul(const float* x, int rows, const kernix_qweight* w, const float* bias, int accumulate, float* y)
{
    const int k = w->k, n = w->n, kp = w->kp, np = w->np;
    int8_t xs[8192] __attribute__((aligned(16)));
    int32_t as[4096] __attribute__((aligned(16)));
    int8_t* xq = kp <= 8192 ? xs : malloc((size_t)kp);
    int32_t* acc = np <= 4096 ? as : malloc(sizeof(int32_t) * (size_t)np);
    if (xq == NULL || acc == NULL) {
        if (xq != xs) free(xq);
        if (acc != as) free(acc);
        return -1;
    }
    memset(xq + k, 0, (size_t)(kp - k)); /* 늘린 칸은 0: 가중치 쪽도 0이지만 쓰레기 값이 NaN을 만들지 않게 */
    for (int r = 0; r < rows; r++) {
        const float* xr = x + (size_t)r * k;
        float* yr = y + (size_t)r * n;
        float m = 0.0f;
        int j = 0;
#if defined(__ARM_FEATURE_DOTPROD)
        float32x4_t mv = vdupq_n_f32(0.0f); /* vmaxq_f32(FMAX)는 NaN을 전파하므로 NaN 행은 m이 NaN이 됨 */
        for (; j + 4 <= k; j += 4) mv = vmaxq_f32(mv, vabsq_f32(vld1q_f32(xr + j)));
        m = vmaxvq_f32(mv);
#endif
        for (; j < k; j++) { const float a = fabsf(xr[j]); m = (a > m || a != a) ? a : m; }
        if (!(m <= FLT_MAX)) { /* NaN 또는 무한대 */
            for (int j = 0; j < n; j++) yr[j] = NAN;
            continue;
        }
        const float sx = m > 0.0f ? m / 127.0f : 1.0f, inv = 1.0f / sx;
        j = 0;
#if defined(__ARM_FEATURE_DOTPROD)
        const int32x4_t lo = vdupq_n_s32(-127), hi = vdupq_n_s32(127);
        for (; j + 16 <= k; j += 16) {
            int32x4_t a[4];
            for (int t = 0; t < 4; t++)
                a[t] = vminq_s32(hi, vmaxq_s32(lo, vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(xr + j + 4 * t), inv))));
            const int16x8_t h0 = vcombine_s16(vmovn_s32(a[0]), vmovn_s32(a[1])), h1 = vcombine_s16(vmovn_s32(a[2]), vmovn_s32(a[3]));
            vst1q_s8(xq + j, vcombine_s8(vmovn_s16(h0), vmovn_s16(h1)));
        }
#endif
        for (; j < k; j++) {
            float v = rintf(xr[j] * inv);
            xq[j] = (int8_t)(v > 127.0f ? 127.0f : v < -127.0f ? -127.0f : v);
        }
#if defined(__ARM_FEATURE_DOTPROD)
        const int8_t* p = w->q;
        for (int nb = 0; nb < np / 16; nb++) {
            int32x4_t a0 = vdupq_n_s32(0), a1 = a0, a2 = a0, a3 = a0;
            for (int kb = 0; kb < kp / 16; kb++) {
                const int8x16_t xv = vld1q_s8(xq + kb * 16);
#define KERNIX_QSTEP(l) \
                a0 = vdotq_laneq_s32(a0, vld1q_s8(p), xv, l);      a1 = vdotq_laneq_s32(a1, vld1q_s8(p + 16), xv, l); \
                a2 = vdotq_laneq_s32(a2, vld1q_s8(p + 32), xv, l); a3 = vdotq_laneq_s32(a3, vld1q_s8(p + 48), xv, l); p += 64;
                KERNIX_QSTEP(0) KERNIX_QSTEP(1) KERNIX_QSTEP(2) KERNIX_QSTEP(3)
#undef KERNIX_QSTEP
            }
            vst1q_s32(acc + nb * 16, a0); vst1q_s32(acc + nb * 16 + 4, a1);
            vst1q_s32(acc + nb * 16 + 8, a2); vst1q_s32(acc + nb * 16 + 12, a3);
        }
#else   /* dotprod 없는 CPU: 같은 결과의 스칼라 */
        for (int c = 0; c < n; c++) {
            int32_t s = 0;
            for (int t = 0; t < k; t++) s += (int32_t)xq[t] * w->q[kernix_qweight_at(w, t, c)];
            acc[c] = s;
        }
#endif
        int c = 0;
#if defined(__ARM_FEATURE_DOTPROD)
        for (; c + 4 <= n; c += 4) { /* 스칼라 꼬리와 같은 연산 순서: (acc·sx)·scale + bias (+ y) */
            float32x4_t v = vmulq_f32(vmulq_n_f32(vcvtq_f32_s32(vld1q_s32(acc + c)), sx), vld1q_f32(w->scale + c));
            if (bias) v = vaddq_f32(v, vld1q_f32(bias + c));
            if (accumulate) v = vaddq_f32(vld1q_f32(yr + c), v);
            vst1q_f32(yr + c, v);
        }
#endif
        for (; c < n; c++) {
            const float v = (float)acc[c] * sx * w->scale[c] + (bias ? bias[c] : 0.0f);
            yr[c] = accumulate ? yr[c] + v : v;
        }
    }
    if (xq != xs) free(xq);
    if (acc != as) free(acc);
    return 0;
}


/* 값 하나 (그래프의 간선). 버퍼에서는 (배치·r) × c */
typedef struct {
    int nd, d[KERNIX_MAX_DIMS]; /* 표본 하나의 모양, channels-last (마지막 축 = 채널) */
    int r, c;                   /* 표본당 행 수(앞 축들의 곱), 열 수(마지막 축) */
    int st;                     /* 저장 버퍼 번호, -1 = 호출자 입력 (읽기 전용) */
    int last;                   /* 이 값을 마지막으로 읽는 단계 번호 (모델 출력이면 n_steps) */
} kernix_val;

typedef struct {
    int kind;            /* KERNIX_OP_* */
    int act;             /* LINEAR/CONV: 합쳐진 활성화 또는 NONE. ACT/RNN: 활성화 */
    float alpha;         /* 활성화 인자 (LeakyReLU 기울기, ELU alpha) */
    int n_in, inp[KERNIX_MAX_INPUTS]; /* 입력 값 번호 (0 = 모델 입력, k+1 = 단계 k의 출력) */
    int in, out;         /* LINEAR in/out, LAYERNORM cols, RNN in/units, CONV cin/cout, ATTENTION 출력 채널 */
    float_tensor* w;     /* LINEAR in×out, RNN 입력 가중치 in×G·units, CONV kh·kw·(cin/groups)×cout, EMBEDDING vocab×dim,
                            ATTENTION query 투영 cq×heads·dk */
    float* bias;         /* LINEAR out개, RNN G·units개 (GRU는 입력·순환 2벌), CONV cout개, ATTENTION query 투영 bias */
    float* gamma;        /* LAYERNORM gamma, AFFINE scale */
    float* beta;         /* LAYERNORM beta, AFFINE shift */
    float eps;
    float_tensor* buf;   /* 출력 값의 텐서 보기 (작업 버퍼가 할당돼 있을 때만, 이전 버전 호환) */
    /* RNN (LSTM 게이트 i f g o, GRU 게이트 z r h: Keras 순서) */
    float* rec;          /* 순환 가중치 units×G·units */
    int seq;             /* 시퀀스 길이 T: 입력 행 = 배치×T (표본마다 T행이 연속) */
    int ret_seq;         /* 1이면 매 시점 출력, 0이면 마지막 시점만 */
    int dirs, merge;     /* 방향 수(1, 2)와 양방향 합치기 KERNIX_MERGE_* */
    float_tensor* w_b;   /* 역방향 가중치들 (dirs == 2) */
    float* rec_b;
    float* bias_b;
    kernix_qweight* q_b;
    float_tensor* deq_b;
    float_tensor* tmp[6]; /* 단계별 작업 버퍼 (연산마다 쓰임이 다름) */
    kernix_qweight* q;    /* int8 가중치: LINEAR면 w를, RNN이면 rec을 대신함 (그때 w 또는 rec은 NULL) */
    float_tensor* deq;    /* q를 풀어 쓴 float 사본 (작업 버퍼): 할당 행 수가 KERNIX_INT8_MAX_ROWS를 넘을 때만 */
    /* CONV / POOL / PAD / UPSAMPLE */
    int kh, kw, sh, sw, dh, dw, pt, pb, pl, pr, groups;
    int ptype, count_pad;  /* POOL: KERNIX_POOL_*, 평균에 패딩 칸을 셀지 (PyTorch count_include_pad) */
    int perm[KERNIX_MAX_DIMS]; /* PERMUTE: 출력 축 k = 입력 축 perm[k] */
    int nd, d[KERNIX_MAX_DIMS]; /* RESHAPE 목표 모양 */
    int axis, start, len, squeeze; /* CONCAT/SELECT */
    int bop;               /* BINARY: KERNIX_BIN_* */
    /* ATTENTION: 입력 (query[, value[, key]]), key가 없으면 value, value가 없으면 query */
    int heads, dk, dv, causal;
    float_tensor* wk;      /* key 투영 ck×heads·dk */
    float_tensor* wv;      /* value 투영 cv×heads·dv */
    float_tensor* wo;      /* 출력 투영 heads·dv×out */
    float* bk;
    float* bv;
    float* bo;
    float* cst;            /* ADD_CONST: 표본 하나 크기의 상수 */
} kernix_step;

/* forward 작업 버퍼(단계별 중간 결과)의 기본 상한. 배치가 이보다 큰 작업 버퍼를 요구하면 표본을 나눠 여러 번 실행해
 * 최대 메모리가 배치 크기와 무관해진다. kernix_model_set_workspace_limit으로 모델마다 바꿀 수 있음 (0 = 제한 없음) */
#ifndef KERNIX_WORKSPACE_LIMIT
#define KERNIX_WORKSPACE_LIMIT ((size_t)64 << 20)
#endif

typedef struct kernix_model {
    int n_steps;
    int in_dim, out_dim;   /* 입력 채널 수, 표본당 출력 float 수 */
    int seq_len;           /* 표본당 입력 행 수: 입력 행 수는 이 값의 배수여야 함 (시퀀스 T, 이미지 H·W, 벡터 1) */
    int batch;             /* 작업 버퍼가 할당된 입력 행 수(한 번에 실행하는 조각 크기), 없으면 -1 */
    kernix_step* steps;
    float_tensor* in_copy; /* 항상 NULL (이전 버전 호환: 입력 복사는 저장 버퍼 계획에 포함됨) */
    float_tensor* out_all; /* 나눠 실행할 때 모은 전체 출력 (배치×out_dim) */
    size_t ws_limit;       /* 작업 버퍼 상한 (바이트), 0 = 제한 없음 */
    size_t ws_per_sample;  /* 표본 하나당 작업 버퍼 float 수 (로드 시 계산) */
    size_t ws_fixed;       /* int8 가중치를 풀어 쓴 사본의 float 수 합 (큰 배치에서만 할당, 조각 계산에서 먼저 뺌) */
    int n_vals;            /* n_steps + 1 */
    int out_v;             /* 모델 출력 값 번호 */
    kernix_val* vals;
    float_tensor* views;   /* views[v]: 지금 조각에서 값 v의 텐서 */
    float_tensor out_view; /* 모델 출력: 배치 × out_dim 보기 */
    int n_st;              /* 저장 버퍼 수 */
    size_t* st_cap;        /* 저장 버퍼마다 표본당 float 수 */
    float_tensor** st;     /* 저장 버퍼 (배치 × st_cap) */
    int in_nd, in_shape[KERNIX_MAX_DIMS];   /* 표본 하나의 입력 모양 (channels-last) */
    int out_nd, out_shape[KERNIX_MAX_DIMS]; /* 표본 하나의 출력 모양 (channels-last) */
    int layout;            /* 0 = channels-last, 1 = PyTorch channels-first 입력 (바인딩이 축을 옮김) */
} kernix_model;

enum { KERNIX_SHAPE_COUNT, KERNIX_SHAPE_ALLOC, KERNIX_SHAPE_VIEW };
#define KERNIX_SHAPE_FAIL ((size_t)-1)
static inline size_t kernix_model_shape(kernix_model* m, int rows, int mode); /* 아래 정의, 로드에서 사용 */

/* 작업 버퍼(저장 버퍼, 단계별 tmp/deq, out_all)를 해제. Metal 버퍼는 재사용 캐시로 가서 재할당이 빠름 */
static inline void kernix_model_release(kernix_model* m)
{
    if (m == NULL) return;
    m->batch = -1;
    float_free(m->out_all);
    m->out_all = NULL;
    for (int k = 0; m->st != NULL && k < m->n_st; k++) {
        float_free(m->st[k]);
        m->st[k] = NULL;
    }
    for (int i = 0; m->steps != NULL && i < m->n_steps; i++) {
        kernix_step* s = &m->steps[i];
        s->buf = NULL;
        for (int k = 0; k < 6; k++) { float_free(s->tmp[k]); s->tmp[k] = NULL; }
        float_free(s->deq);
        s->deq = NULL;
        float_free(s->deq_b);
        s->deq_b = NULL;
    }
}

/* 작업 버퍼를 운영체제에 반환 (가중치는 유지, 다음 forward가 다시 할당). 큰 배치 뒤 쉬는 동안 부름.
 * Metal 빌드에서는 프로세스 전체의 Metal 재사용 캐시도 비움 (다른 모델의 다음 할당이 조금 느려질 수 있음) */
static inline void kernix_model_trim(kernix_model* m)
{
    kernix_model_release(m);
#if defined(KERNIX_USE_METAL)
    kernix_metal_release_cache();
#endif
}

static inline void kernix_step_free(kernix_step* s)
{
    float_free(s->w); float_free(s->w_b); float_free(s->wk); float_free(s->wv); float_free(s->wo);
    kernix_qweight_free(s->q); kernix_qweight_free(s->q_b);
    free(s->rec); free(s->rec_b); free(s->bias); free(s->bias_b); free(s->bk); free(s->bv); free(s->bo);
    free(s->gamma); free(s->beta); free(s->cst);
}

static inline void kernix_model_free(kernix_model* m)
{
    if (m == NULL) return;
    kernix_model_release(m);
    for (int i = 0; m->steps != NULL && i < m->n_steps; i++) kernix_step_free(&m->steps[i]);
    free(m->steps);
    free(m->vals);
    free(m->views);
    free(m->st_cap);
    free(m->st);
    free(m);
}

/* 파일은 신뢰할 수 없는 입력: 읽기 전에 항상 남은 바이트와 비교 */
typedef struct { FILE* f; size_t left; } kernix_reader;

static inline int kernix_read_u32(kernix_reader* r, uint32_t* v)
{
    if (r->left < 4 || fread(v, 4, 1, r->f) != 1) return -1;
    r->left -= 4;
    return 0;
}

static inline int kernix_read_floats(kernix_reader* r, float* dst, size_t n)
{
    if (dst == NULL || n > r->left / 4 || fread(dst, 4, n, r->f) != n) return -1;
    r->left -= n * 4;
    return 0;
}

/* n개를 새로 할당해 읽음. 파일에 n개가 없으면 할당하지 않고 NULL */
static inline float* kernix_read_alloc(kernix_reader* r, size_t n)
{
    if (n > r->left / 4) return NULL;
    float* p = malloc(n * sizeof(float));
    if (kernix_read_floats(r, p, n) != 0) { free(p); return NULL; }
    return p;
}

/* .knx v2 가중치 블록: uint32 dtype 뒤에 값. F32 = float32 rows×cols, I8_COL = float32 scale[cols] + int8 rows×cols,
 * I8_ROW = float32 scale[rows] + int8 rows×cols (w ≈ q·scale). v1 파일은 dtype 없이 F32 */
enum { KERNIX_DT_F32, KERNIX_DT_I8_COL, KERNIX_DT_I8_ROW };

/* 블록 본문의 바이트 수 (dtype 4바이트 제외). 할당 전 크기 확인용 */
static inline uint64_t kernix_weight_bytes(uint32_t dt, uint64_t k, uint64_t n)
{
    return dt == KERNIX_DT_F32 ? 4 * k * n : dt == KERNIX_DT_I8_COL ? 4 * n + k * n : 4 * k + k * n;
}

/* v2면 dtype을 읽음 (v1은 F32). 알 수 없는 dtype은 거부 */
static inline int kernix_read_dtype(kernix_reader* r, int v2, uint32_t* dt, const char** err)
{
    *dt = KERNIX_DT_F32;
    if (v2 && kernix_read_u32(r, dt) != 0) return -1;
    if (*dt > KERNIX_DT_I8_ROW) { *err = "알 수 없는 가중치 형식"; return -1; }
    return 0;
}

/* k×n 가중치 본문을 읽음. keep_int8이고 I8_COL이면 *wq(int8 묶음)로, 그 밖에는 dst(float k×n, 호출자가 할당)로.
 * 호출자는 kernix_weight_bytes로 남은 바이트를 먼저 확인해야 함 */
static inline int kernix_read_weight(kernix_reader* r, uint32_t dt, int k, int n, int keep_int8,
                                     float* dst, kernix_qweight** wq, const char** err)
{
    const size_t cnt = (size_t)k * (size_t)n;
    if (dt == KERNIX_DT_F32) return kernix_read_floats(r, dst, cnt);
    if (keep_int8 && dt == KERNIX_DT_I8_COL && k > KERNIX_INT8_MAX_K) { *err = "int8 가중치 입력 크기가 범위 밖"; return -1; }
    const size_t ns = dt == KERNIX_DT_I8_COL ? (size_t)n : (size_t)k;
    float* sc = malloc(sizeof(float) * ns);
    int8_t* q = malloc(cnt);
    int rc = -1;
    if (sc == NULL || q == NULL) { *err = "메모리 부족"; goto done; }
    if (kernix_read_floats(r, sc, ns) != 0 || cnt > r->left || fread(q, 1, cnt, r->f) != cnt) goto done;
    r->left -= cnt;
    for (size_t i = 0; i < ns; i++)
        if (!(sc[i] > 0.0f && sc[i] <= FLT_MAX)) { *err = "int8 scale이 0 이하이거나 유한하지 않음"; goto done; }
    if (keep_int8 && dt == KERNIX_DT_I8_COL) {
        if ((*wq = kernix_qweight_pack(q, sc, k, n)) == NULL) { *err = "메모리 부족"; goto done; }
    } else {
        for (size_t i = 0; i < (size_t)k; i++)
            for (size_t j = 0; j < (size_t)n; j++) dst[i * n + j] = (float)q[i * n + j] * sc[dt == KERNIX_DT_I8_COL ? j : i];
    }
    rc = 0;
done:
    free(sc);
    free(q);
    return rc;
}

/* ---- 활성화 ---- */
static inline float kernix_relu6f(float v) { return v < 0.0f ? 0.0f : v > 6.0f ? 6.0f : v; }
static inline float kernix_softplusf(float v) { return v > 20.0f ? v : log1pf(expf(v)); }

/* 활성화를 제자리로 적용. len ≤ KERNIX_CHUNK. 0–4번 수식은 kernix.h의 floats_relu/gelu/silu/sigmoid/tanh와 같고,
 * 모든 수식은 src/kernix_metal.m의 bias_act 커널(GPU 후처리)과 같다 */
static inline void kernix_act_chunk(float* o, size_t len, int act, float alpha)
{
    const int n = (int)len;
    const float one = 1.0f, zero = 0.0f;
    float t[KERNIX_CHUNK];
    switch (act) {
    case KERNIX_ACT_RELU: vDSP_vthr(o, 1, &zero, o, 1, len); break;
    case KERNIX_ACT_TANH: vvtanhf(o, o, &n); break;
    case KERNIX_ACT_SIGMOID:
        vDSP_vneg(o, 1, o, 1, len); vvexpf(o, o, &n);
        vDSP_vsadd(o, 1, &one, o, 1, len); vvrecf(o, o, &n);
        break;
    case KERNIX_ACT_SILU: /* x * sigmoid(x): 원본 x가 필요해 시그모이드는 임시 버퍼에 */
        vDSP_vneg(o, 1, t, 1, len); vvexpf(t, t, &n);
        vDSP_vsadd(t, 1, &one, t, 1, len); vvrecf(t, t, &n);
        vDSP_vmul(o, 1, t, 1, o, 1, len);
        break;
    case KERNIX_ACT_GELU:
        for (size_t i = 0; i < len; i++) { float z = o[i] * 0.70710678f; t[i] = -z * z; }
        vvexpf(t, t, &n);
        for (size_t i = 0; i < len; i++) {
            float x = o[i], z = fabsf(x) * 0.70710678f;
            float k = 1.0f / (1.0f + 0.3275911f * z);
            float p = k * (0.254829592f + k * (-0.284496736f + k * (1.421413741f
                    + k * (-1.453152027f + k * 1.061405429f))));
            float e = 1.0f - p * t[i];
            o[i] = 0.5f * x * (1.0f + (x < 0.0f ? -e : e));
        }
        break;
    case KERNIX_ACT_LEAKY_RELU: for (size_t i = 0; i < len; i++) o[i] = o[i] < 0.0f ? alpha * o[i] : o[i]; break;
    case KERNIX_ACT_ELU: for (size_t i = 0; i < len; i++) o[i] = o[i] < 0.0f ? alpha * expm1f(o[i]) : o[i]; break;
    case KERNIX_ACT_SELU:
        for (size_t i = 0; i < len; i++) o[i] = 1.0507009873554805f * (o[i] < 0.0f ? 1.6732632423543772f * expm1f(o[i]) : o[i]);
        break;
    case KERNIX_ACT_SOFTPLUS: for (size_t i = 0; i < len; i++) o[i] = kernix_softplusf(o[i]); break;
    case KERNIX_ACT_RELU6: for (size_t i = 0; i < len; i++) o[i] = kernix_relu6f(o[i]); break;
    case KERNIX_ACT_HARD_SIGMOID: for (size_t i = 0; i < len; i++) o[i] = kernix_relu6f(o[i] + 3.0f) / 6.0f; break;
    case KERNIX_ACT_HARD_SWISH: for (size_t i = 0; i < len; i++) o[i] = o[i] * kernix_relu6f(o[i] + 3.0f) / 6.0f; break;
    case KERNIX_ACT_GELU_TANH:
        for (size_t i = 0; i < len; i++) {
            const float x = o[i];
            o[i] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
        }
        break;
    case KERNIX_ACT_EXP: vvexpf(o, o, &n); break;
    case KERNIX_ACT_MISH: for (size_t i = 0; i < len; i++) o[i] = o[i] * tanhf(kernix_softplusf(o[i])); break;
    case KERNIX_ACT_SOFTSIGN: for (size_t i = 0; i < len; i++) o[i] = o[i] / (1.0f + fabsf(o[i])); break;
    default: break;
    }
}

/* 행 하나(길이 cols)에 활성화. KERNIX_CHUNK 단위로 나눠 처리 */
static inline void kernix_act_row(float* o, size_t cols, int act, float alpha)
{
    if (act == KERNIX_ACT_NONE) return;
    for (size_t c = 0; c < cols; c += KERNIX_CHUNK) kernix_act_chunk(o + c, cols - c < KERNIX_CHUNK ? cols - c : KERNIX_CHUNK, act, alpha);
}

/* buf(rows×cols)의 각 행에 bias(cols개, NULL 가능)를 더하고 활성화를 제자리 적용.
 * 32KB 조각 단위라 더하기와 활성화가 캐시 안에서 한 번에 끝남 */
static inline void kernix_bias_act(float* buf, size_t rows, size_t cols, const float* bias, int act, float alpha)
{
    if (bias == NULL && act == KERNIX_ACT_NONE) return;
    kernix_par(rows, cols, ^(size_t r0, size_t nr) {
        for (size_t r = r0; r < r0 + nr; r++) {
            for (size_t c = 0; c < cols; c += KERNIX_CHUNK) {
                size_t len = cols - c < KERNIX_CHUNK ? cols - c : KERNIX_CHUNK;
                float* o = buf + r * cols + c;
                if (bias != NULL) vDSP_vadd(o, 1, bias + c, 1, o, 1, len);
                kernix_act_chunk(o, len, act, alpha);
            }
        }
    });
}

/* y = act(a·W + bias): GPU가 맡은 행은 GPU에서 bias·act까지, CPU 행은 GPU를 기다리는 동안 CPU에서 */
static inline int kernix_dense(const float_tensor* a, const float_tensor* W, float_tensor* y, const float* bias, int act, float alpha)
{
    float* o = y->tensor;
    const size_t out = (size_t)y->col;
    return kernix_matmul_ep_into(a, W, y, bias, act, alpha,
        ^(size_t r0, size_t r1) { kernix_bias_act(o + r0 * out, r1 - r0, out, bias, act, alpha); });
}

static inline void kernix_copy_if(const float_tensor* a, float_tensor* y)
{
    if (a->tensor != y->tensor) memcpy(y->tensor, a->tensor, sizeof(float) * (size_t)a->row * (size_t)a->col);
}

/* ---- RNN (LSTM / GRU / SimpleRNN) ---- */
static inline size_t kernix_gates(int kind) { return kind == KERNIX_OP_LSTM ? 4 : kind == KERNIX_OP_GRU ? 3 : 1; }

/* 한 방향 실행. x: 배치·T × in. 입력 투영(x·W)은 모든 시점을 한 번의 행렬 곱으로 계산하고, 시점마다 h·U만 더한 뒤
 * 게이트를 제자리로 처리. 출력: ret_seq면 시점 t의 은닉 상태를 dst + (b·T + t)·stride에, 아니면 마지막 상태를 dst + b·stride에.
 * 역방향(dir 1)은 t = T−1부터 거꾸로 읽고, 출력은 같은 시점 위치에 둔다 (Keras Bidirectional, PyTorch와 같음) */
static inline int kernix_rnn_dir(const kernix_step* s, int dir, const float_tensor* x, float* dst, size_t stride)
{
    const int kind = s->kind, act = s->act, ret_seq = s->ret_seq;
    const size_t T = (size_t)s->seq, u = (size_t)s->out, gu = kernix_gates(kind) * u, B = (size_t)x->row / T;
    if (B == 0) return 0;
    const float_tensor* W = dir ? s->w_b : s->w;
    const kernix_qweight* q = dir ? s->q_b : s->q;
    const float_tensor* deq = dir ? s->deq_b : s->deq;
    const float* bias = dir ? s->bias_b : s->bias;
    const float* brec = kind == KERNIX_OP_GRU ? bias + gu : NULL; /* GRU: 순환 쪽 bias (reset_after) */
    const float* U = q != NULL ? (deq != NULL ? deq->tensor : NULL) : (dir ? s->rec_b : s->rec);
    float* zx = s->tmp[0]->tensor;
    float* z = s->tmp[1]->tensor;
    float* c = kind == KERNIX_OP_LSTM ? s->tmp[2]->tensor : NULL;
    float* h = s->tmp[3]->tensor;
    float* zh = kind == KERNIX_OP_GRU ? s->tmp[4]->tensor : NULL;
    if (kernix_dense(x, W, s->tmp[0], bias, KERNIX_ACT_NONE, 0.0f) != 0) return -1;
    for (size_t step = 0; step < T; step++) {
        const size_t t = dir ? T - 1 - step : step;
        for (size_t b = 0; b < B; b++) memcpy(z + b * gu, zx + (b * T + t) * gu, sizeof(float) * gu);
        float* acc = kind == KERNIX_OP_GRU ? zh : z; /* GRU는 h·U를 따로 둠: r이 후보 쪽 순환 항에만 곱해짐 */
        if (step == 0) {
            if (kind == KERNIX_OP_GRU) for (size_t b = 0; b < B; b++) memcpy(zh + b * gu, brec, sizeof(float) * gu); /* h = 0 */
        } else if (q != NULL && B <= KERNIX_INT8_MAX_ROWS) { /* int8이면 행(배치)이 적을 때 sdot, 많으면 풀어 둔 사본 */
            if (kernix_qmatmul(h, (int)B, q, brec, kind != KERNIX_OP_GRU, acc) != 0) return -1;
        } else {
            if (U == NULL) return -1;
            if (kind == KERNIX_OP_GRU) for (size_t b = 0; b < B; b++) memcpy(zh + b * gu, brec, sizeof(float) * gu);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)B, (int)gu, (int)u,
                        1.0f, h, (int)u, U, (int)gu, 1.0f, acc, (int)gu);
        }
        kernix_par(B, 6 * u, ^(size_t b0, size_t nb) {
            for (size_t b = b0; b < b0 + nb; b++) {
                float* zb = z + b * gu;
                float* hb = h + b * u;
                if (kind == KERNIX_OP_LSTM) { /* i, f, g, o */
                    float* cb = c + b * u;
                    kernix_act_chunk(zb, 2 * u, KERNIX_ACT_SIGMOID, 0.0f);
                    kernix_act_chunk(zb + 2 * u, u, act, s->alpha);
                    kernix_act_chunk(zb + 3 * u, u, KERNIX_ACT_SIGMOID, 0.0f);
                    if (step == 0) { /* c = i·g. 이전 호출의 셀 상태가 버퍼에 남아 있으므로 덮어씀 */
                        vDSP_vmul(zb, 1, zb + 2 * u, 1, cb, 1, u);
                    } else {         /* c = f·c + i·g */
                        vDSP_vmul(zb + u, 1, cb, 1, cb, 1, u);
                        vDSP_vma(zb, 1, zb + 2 * u, 1, cb, 1, cb, 1, u);
                    }
                    memcpy(hb, cb, sizeof(float) * u); /* h = o·act(c) */
                    kernix_act_chunk(hb, u, act, s->alpha);
                    vDSP_vmul(hb, 1, zb + 3 * u, 1, hb, 1, u);
                } else if (kind == KERNIX_OP_GRU) { /* z, r, h~ */
                    const float* hz = zh + b * gu;
                    float* hh = zb + 2 * u;
                    vDSP_vadd(zb, 1, hz, 1, zb, 1, 2 * u);
                    kernix_act_chunk(zb, 2 * u, KERNIX_ACT_SIGMOID, 0.0f);
                    vDSP_vma(zb + u, 1, hz + 2 * u, 1, hh, 1, hh, 1, u); /* h~ = act(x_h + r·(h·U_h + b_h)) */
                    kernix_act_chunk(hh, u, act, s->alpha);
                    if (step == 0) vDSP_vneg(hh, 1, hb, 1, u);         /* 이전 h = 0 */
                    else vDSP_vsub(hh, 1, hb, 1, hb, 1, u);            /* h − h~ */
                    vDSP_vma(hb, 1, zb, 1, hh, 1, hb, 1, u);           /* h = z·(h − h~) + h~ */
                } else {
                    memcpy(hb, zb, sizeof(float) * u);
                    kernix_act_chunk(hb, u, act, s->alpha);
                }
                if (ret_seq) memcpy(dst + (b * T + t) * stride, hb, sizeof(float) * u);
            }
        });
    }
    if (!ret_seq) for (size_t b = 0; b < B; b++) memcpy(dst + b * stride, h + b * u, sizeof(float) * u);
    return 0;
}

static inline int kernix_rnn_run(const kernix_step* s, const float_tensor* x, float_tensor* y)
{
    const size_t u = (size_t)s->out;
    if (s->dirs == 1) return kernix_rnn_dir(s, 0, x, y->tensor, u);
    if (s->merge == KERNIX_MERGE_CONCAT)
        return kernix_rnn_dir(s, 0, x, y->tensor, 2 * u) != 0 || kernix_rnn_dir(s, 1, x, y->tensor + u, 2 * u) != 0 ? -1 : 0;
    float* bw = s->tmp[5]->tensor;
    if (kernix_rnn_dir(s, 0, x, y->tensor, u) != 0 || kernix_rnn_dir(s, 1, x, bw, u) != 0) return -1;
    const size_t n = (size_t)y->row * (size_t)y->col;
    const float half = 0.5f;
    if (s->merge == KERNIX_MERGE_MUL) vDSP_vmul(y->tensor, 1, bw, 1, y->tensor, 1, n);
    else vDSP_vadd(y->tensor, 1, bw, 1, y->tensor, 1, n);
    if (s->merge == KERNIX_MERGE_AVE) vDSP_vsmul(y->tensor, 1, &half, y->tensor, 1, n);
    return 0;
}

/* ---- 합성곱 · 풀링 ---- */
/* 공간 크기: nd 2(T×C)는 H = T, W = 1, nd 3(H×W×C)은 그대로 */
static inline void kernix_hw(const kernix_val* v, int* H, int* W)
{
    *H = v->d[0];
    *W = v->nd == 3 ? v->d[1] : 1;
}

/* 출력 행 (b, oy, ox)마다 커널 창의 입력 채널 [c0, c0+cg)를 (ky, kx, c) 순서로 펼침. 범위 밖(패딩)은 0 */
static inline void kernix_im2col(const kernix_step* s, const float* x, float* col, size_t B, int H, int W, int C,
                                 int c0, int cg, int Ho, int Wo)
{
    const size_t K = (size_t)s->kh * (size_t)s->kw * (size_t)cg;
    kernix_par(B * (size_t)Ho * (size_t)Wo, K, ^(size_t r0, size_t nr) {
        for (size_t r = r0; r < r0 + nr; r++) {
            const size_t b = r / ((size_t)Ho * Wo), oy = (r / Wo) % Ho, ox = r % Wo;
            float* o = col + r * K;
            for (int ky = 0; ky < s->kh; ky++) {
                const long iy = (long)oy * s->sh - s->pt + (long)ky * s->dh;
                for (int kx = 0; kx < s->kw; kx++, o += cg) {
                    const long ix = (long)ox * s->sw - s->pl + (long)kx * s->dw;
                    if (iy < 0 || iy >= H || ix < 0 || ix >= W) memset(o, 0, sizeof(float) * cg);
                    else memcpy(o, x + (((size_t)b * H + iy) * W + ix) * C + c0, sizeof(float) * cg);
                }
            }
        }
    });
}

static inline int kernix_conv_pointwise(const kernix_step* s)
{
    return s->kh == 1 && s->kw == 1 && s->sh == 1 && s->sw == 1 && s->pt == 0 && s->pb == 0 && s->pl == 0 && s->pr == 0;
}

/* groups == 1: im2col + 행렬곱 하나 (1×1 보폭 1은 입력을 그대로). 깊이별(groups == cin): 채널 방향 벡터 곱-합.
 * 그 밖의 groups: 그룹마다 im2col + sgemm을 출력 열 위치에 바로 씀 */
static inline int kernix_conv_run(const kernix_step* s, const kernix_val* vi, const float_tensor* x,
                                  const kernix_val* vo, float_tensor* y)
{
    int H, W, Ho, Wo;
    kernix_hw(vi, &H, &W);
    kernix_hw(vo, &Ho, &Wo);
    const int C = vi->c, Co = vo->c, g = s->groups, cg = C / g, cog = Co / g;
    const size_t B = (size_t)x->row / ((size_t)H * W), rows = B * (size_t)Ho * Wo;
    const size_t K = (size_t)s->kh * s->kw * cg;
    if (B == 0) return 0;
    const float* bias = s->bias;
    const int act = s->act;
    const float alpha = s->alpha;
    float* o = y->tensor;
    if (g == 1) {
        if (kernix_conv_pointwise(s)) return kernix_dense(x, s->w, y, bias, act, alpha);
        kernix_im2col(s, x->tensor, s->tmp[0]->tensor, B, H, W, C, 0, C, Ho, Wo);
        return kernix_dense(s->tmp[0], s->w, y, bias, act, alpha);
    }
    if (cg == 1) { /* 깊이별: 출력 채널 c·mult + m은 입력 채널 c에서 */
        const int mult = Co / C;
        const float* w = s->w->tensor;
        const float* xin = x->tensor;
        kernix_par(rows, (size_t)s->kh * s->kw * Co, ^(size_t r0, size_t nr) {
            for (size_t r = r0; r < r0 + nr; r++) {
                const size_t b = r / ((size_t)Ho * Wo), oy = (r / Wo) % Ho, ox = r % Wo;
                float* out = o + r * Co;
                if (bias != NULL) memcpy(out, bias, sizeof(float) * Co);
                else memset(out, 0, sizeof(float) * Co);
                for (int ky = 0; ky < s->kh; ky++) {
                    const long iy = (long)oy * s->sh - s->pt + (long)ky * s->dh;
                    if (iy < 0 || iy >= H) continue;
                    for (int kx = 0; kx < s->kw; kx++) {
                        const long ix = (long)ox * s->sw - s->pl + (long)kx * s->dw;
                        if (ix < 0 || ix >= W) continue;
                        const float* xp = xin + (((size_t)b * H + iy) * W + ix) * C;
                        const float* wt = w + ((size_t)ky * s->kw + kx) * Co;
                        if (mult == 1) vDSP_vma(xp, 1, wt, 1, out, 1, out, 1, (size_t)C);
                        else for (int ci = 0; ci < C; ci++) for (int mi = 0; mi < mult; mi++) out[ci * mult + mi] += xp[ci] * wt[ci * mult + mi];
                    }
                }
                kernix_act_row(out, (size_t)Co, act, alpha);
            }
        });
        return 0;
    }
    float* col = s->tmp[0]->tensor;
    for (int gi = 0; gi < g; gi++) {
        kernix_im2col(s, x->tensor, col, B, H, W, C, gi * cg, cg, Ho, Wo);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)rows, cog, (int)K,
                    1.0f, col, (int)K, s->w->tensor + (size_t)gi * cog, Co, 0.0f, o + (size_t)gi * cog, Co);
    }
    kernix_bias_act(o, rows, (size_t)Co, bias, act, alpha);
    return 0;
}

static inline void kernix_pooling_run(const kernix_step* s, const kernix_val* vi, const float_tensor* x,
                                   const kernix_val* vo, float_tensor* y)
{
    int H, W, Ho, Wo;
    kernix_hw(vi, &H, &W);
    kernix_hw(vo, &Ho, &Wo);
    const size_t C = (size_t)vi->c, B = (size_t)x->row / ((size_t)H * W);
    const float* xin = x->tensor;
    float* o = y->tensor;
    kernix_par(B * (size_t)Ho * Wo, (size_t)s->kh * s->kw * C, ^(size_t r0, size_t nr) {
        for (size_t r = r0; r < r0 + nr; r++) {
            const size_t b = r / ((size_t)Ho * Wo), oy = (r / Wo) % Ho, ox = r % Wo;
            float* out = o + r * C;
            const float init = s->ptype == KERNIX_POOL_MAX ? -INFINITY : 0.0f;
            vDSP_vfill(&init, out, 1, C);
            int cnt = 0;
            for (int ky = 0; ky < s->kh; ky++) {
                const long iy = (long)oy * s->sh - s->pt + ky;
                if (iy < 0 || iy >= H) continue;
                for (int kx = 0; kx < s->kw; kx++) {
                    const long ix = (long)ox * s->sw - s->pl + kx;
                    if (ix < 0 || ix >= W) continue;
                    const float* xp = xin + (((size_t)b * H + iy) * W + ix) * C;
                    if (s->ptype == KERNIX_POOL_MAX) vDSP_vmax(out, 1, xp, 1, out, 1, C);
                    else vDSP_vadd(out, 1, xp, 1, out, 1, C);
                    cnt++;
                }
            }
            if (s->ptype == KERNIX_POOL_AVG) {
                const float inv = 1.0f / (float)(s->count_pad ? s->kh * s->kw : (cnt ? cnt : 1));
                vDSP_vsmul(out, 1, &inv, out, 1, C);
            }
        }
    });
}

/* 표본마다 r행을 하나로 (평균 또는 최댓값) */
static inline void kernix_global_pool_run(const kernix_step* s, const kernix_val* vi, const float_tensor* x, float_tensor* y)
{
    const size_t R = (size_t)vi->r, C = (size_t)vi->c, B = (size_t)x->row / R;
    const float* xin = x->tensor;
    float* o = y->tensor;
    kernix_par(B, R * C, ^(size_t b0, size_t nb) {
        for (size_t b = b0; b < b0 + nb; b++) {
            float* out = o + b * C;
            memcpy(out, xin + b * R * C, sizeof(float) * C);
            for (size_t i = 1; i < R; i++) {
                const float* xp = xin + (b * R + i) * C;
                if (s->ptype == KERNIX_POOL_MAX) vDSP_vmax(out, 1, xp, 1, out, 1, C);
                else vDSP_vadd(out, 1, xp, 1, out, 1, C);
            }
            if (s->ptype == KERNIX_POOL_AVG) {
                const float inv = 1.0f / (float)R;
                vDSP_vsmul(out, 1, &inv, out, 1, C);
            }
        }
    });
}

/* ---- 모양 바꾸기 · 합치기 · 고르기 ---- */
static inline void kernix_permute_run(const kernix_step* s, const kernix_val* vi, const float_tensor* x, float_tensor* y)
{
    const int nd = vi->nd;
    size_t n = 1, ist[KERNIX_MAX_DIMS], od[KERNIX_MAX_DIMS];
    for (int k = nd - 1; k >= 0; k--) { ist[k] = n; n *= (size_t)vi->d[k]; }
    for (int k = 0; k < nd; k++) od[k] = (size_t)vi->d[s->perm[k]];
    const size_t B = n ? (size_t)x->row * (size_t)x->col / n : 0;
    const float* xin = x->tensor;
    float* o = y->tensor;
    const int* perm = s->perm;
    const size_t* istp = ist; /* 블록은 배열을 직접 잡을 수 없음: 포인터로 (호출이 끝날 때까지 유효) */
    const size_t* odp = od;
    if (nd == 2 && perm[0] == 1) { /* 2차원 전치 */
        kernix_par(B, n, ^(size_t b0, size_t nb) {
            for (size_t b = b0; b < b0 + nb; b++) vDSP_mtrans(xin + b * n, 1, o + b * n, 1, odp[0], odp[1]);
        });
        return;
    }
    kernix_par(B, n, ^(size_t b0, size_t nb) {
        for (size_t b = b0; b < b0 + nb; b++) {
            const float* xs = xin + b * n;
            float* os = o + b * n;
            for (size_t i = 0; i < n; i++) {
                size_t rem = i, src = 0;
                for (int k = nd - 1; k >= 0; k--) { src += (rem % odp[k]) * istp[perm[k]]; rem /= odp[k]; }
                os[i] = xs[src];
            }
        }
    });
}

static inline float kernix_bin1(int op, float a, float b)
{
    switch (op) {
    case KERNIX_BIN_ADD: return a + b;
    case KERNIX_BIN_SUB: return a - b;
    case KERNIX_BIN_MUL: return a * b;
    case KERNIX_BIN_DIV: return a / b;
    case KERNIX_BIN_MAX: return a > b ? a : b;
    default: return a < b ? a : b;
    }
}

/* 원소별 이항 연산. 표본 안에서 행(r) 또는 열(c) 방향 크기 1은 늘려 맞춤 (브로드캐스트) */
static inline void kernix_binary_run(const kernix_step* s, const kernix_val* va, const float_tensor* a,
                                     const kernix_val* vb, const float_tensor* b, const kernix_val* vo, float_tensor* y)
{
    const size_t R = (size_t)vo->r, C = (size_t)vo->c, B = (size_t)y->row / R;
    const size_t ra = (size_t)va->r, ca = (size_t)va->c, rb = (size_t)vb->r, cb = (size_t)vb->c;
    const float* ap = a->tensor;
    const float* bp = b->tensor;
    float* o = y->tensor;
    const int op = s->bop;
    kernix_par(B * R, C, ^(size_t i0, size_t ni) {
        for (size_t i = i0; i < i0 + ni; i++) {
            const size_t bi = i / R, ri = i % R;
            const float* ar = ap + (bi * ra + (ra == 1 ? 0 : ri)) * ca;
            const float* br = bp + (bi * rb + (rb == 1 ? 0 : ri)) * cb;
            float* orow = o + i * C;
            if (ca == C && cb == C) {
                switch (op) {
                case KERNIX_BIN_ADD: vDSP_vadd(ar, 1, br, 1, orow, 1, C); break;
                case KERNIX_BIN_SUB: vDSP_vsub(br, 1, ar, 1, orow, 1, C); break;
                case KERNIX_BIN_MUL: vDSP_vmul(ar, 1, br, 1, orow, 1, C); break;
                case KERNIX_BIN_DIV: vDSP_vdiv(br, 1, ar, 1, orow, 1, C); break;
                case KERNIX_BIN_MAX: vDSP_vmax(ar, 1, br, 1, orow, 1, C); break;
                default: vDSP_vmin(ar, 1, br, 1, orow, 1, C); break;
                }
            } else {
                for (size_t j = 0; j < C; j++) orow[j] = kernix_bin1(op, ar[ca == 1 ? 0 : j], br[cb == 1 ? 0 : j]);
            }
        }
    });
}

static inline void kernix_concat_run(const kernix_step* s, const kernix_model* m, const kernix_val* vo, float_tensor* y)
{
    const size_t R = (size_t)vo->r, C = (size_t)vo->c, B = (size_t)y->row / R, per = R * C;
    float* o = y->tensor;
    if (s->axis == KERNIX_AXIS_LAST) {
        size_t off = 0;
        for (int k = 0; k < s->n_in; k++) {
            const float* src = m->views[s->inp[k]].tensor;
            const size_t ck = (size_t)m->vals[s->inp[k]].c;
            kernix_par(B * R, ck, ^(size_t r0, size_t nr) {
                for (size_t r = r0; r < r0 + nr; r++) memcpy(o + r * C + off, src + r * ck, sizeof(float) * ck);
            });
            off += ck;
        }
        return;
    }
    size_t off = 0;
    for (int k = 0; k < s->n_in; k++) {
        const kernix_val* v = &m->vals[s->inp[k]];
        const float* src = m->views[s->inp[k]].tensor;
        const size_t pk = (size_t)v->r * (size_t)v->c;
        for (size_t b = 0; b < B; b++) memcpy(o + b * per + off, src + b * pk, sizeof(float) * pk);
        off += pk;
    }
}

/* 첫 축이면 표본마다 [start, start+len) 블록, 마지막 축이면 행마다 열 [start, start+len) */
static inline void kernix_select_run(const kernix_step* s, const kernix_val* vi, const float_tensor* x, float_tensor* y)
{
    const size_t per = (size_t)vi->r * (size_t)vi->c, B = (size_t)x->row / (size_t)vi->r;
    const float* xin = x->tensor;
    float* o = y->tensor;
    if (s->axis == KERNIX_AXIS_FIRST) {
        const size_t blk = per / (size_t)vi->d[0], n = blk * (size_t)s->len;
        for (size_t b = 0; b < B; b++) memcpy(o + b * n, xin + b * per + (size_t)s->start * blk, sizeof(float) * n);
        return;
    }
    const size_t C = (size_t)vi->c, L = (size_t)s->len, rows = B * (size_t)vi->r;
    kernix_par(rows, L, ^(size_t r0, size_t nr) {
        for (size_t r = r0; r < r0 + nr; r++) memcpy(o + r * L, xin + r * C + s->start, sizeof(float) * L);
    });
}

static inline void kernix_pad_run(const kernix_step* s, const kernix_val* vi, const float_tensor* x, const kernix_val* vo, float_tensor* y)
{
    int H, W, Ho, Wo;
    kernix_hw(vi, &H, &W);
    kernix_hw(vo, &Ho, &Wo);
    const size_t C = (size_t)vi->c, B = (size_t)x->row / ((size_t)H * W);
    const float* xin = x->tensor;
    float* o = y->tensor;
    kernix_par(B * (size_t)Ho, (size_t)Wo * C, ^(size_t r0, size_t nr) {
        for (size_t r = r0; r < r0 + nr; r++) {
            const size_t b = r / Ho;
            const long iy = (long)(r % Ho) - s->pt;
            float* out = o + r * Wo * C;
            memset(out, 0, sizeof(float) * Wo * C);
            if (iy >= 0 && iy < H) memcpy(out + (size_t)s->pl * C, xin + ((size_t)b * H + iy) * W * C, sizeof(float) * W * C);
        }
    });
}

static inline void kernix_upsample_run(const kernix_step* s, const kernix_val* vi, const float_tensor* x, const kernix_val* vo, float_tensor* y)
{
    int H, W, Ho, Wo;
    kernix_hw(vi, &H, &W);
    kernix_hw(vo, &Ho, &Wo);
    const size_t C = (size_t)vi->c, B = (size_t)x->row / ((size_t)H * W);
    const float* xin = x->tensor;
    float* o = y->tensor;
    kernix_par(B * (size_t)Ho * Wo, C, ^(size_t r0, size_t nr) {
        for (size_t r = r0; r < r0 + nr; r++) {
            const size_t b = r / ((size_t)Ho * Wo), oy = (r / Wo) % Ho, ox = r % Wo;
            memcpy(o + r * C, xin + (((size_t)b * H + oy / s->sh) * W + ox / s->sw) * C, sizeof(float) * C);
        }
    });
}

/* id(float로 받은 정수)마다 표 한 행. 범위 밖이거나 정수가 아니면 NaN 행 */
static inline void kernix_embedding_run(const kernix_step* s, const float_tensor* x, float_tensor* y)
{
    const size_t n = (size_t)x->row * (size_t)x->col, D = (size_t)s->out, V = (size_t)s->in;
    const float* ids = x->tensor;
    const float* table = s->w->tensor;
    float* o = y->tensor;
    kernix_par(n, D, ^(size_t i0, size_t ni) {
        for (size_t i = i0; i < i0 + ni; i++) {
            const float id = ids[i];
            if (id >= 0.0f && id < (float)V && id == floorf(id)) memcpy(o + i * D, table + (size_t)id * D, sizeof(float) * D);
            else for (size_t j = 0; j < D; j++) o[i * D + j] = NAN;
        }
    });
}

/* ---- 어텐션 ---- */
/* 멀티헤드 어텐션 (Keras MultiHeadAttention, PyTorch nn.MultiheadAttention).
 * Q = q·Wq + bq, K = k·Wk + bk, V = v·Wv + bv. 헤드마다 softmax(Q_h·K_hᵀ / √dk [+ 인과 마스크])·V_h, 이어 붙여 ·Wo + bo */
static inline int kernix_attention_run(const kernix_step* s, const kernix_model* m, float_tensor* y)
{
    const float_tensor* q = &m->views[s->inp[0]];
    const float_tensor* v = &m->views[s->inp[s->n_in > 1 ? 1 : 0]];
    const float_tensor* k = &m->views[s->inp[s->n_in > 2 ? 2 : s->n_in > 1 ? 1 : 0]];
    const size_t Rq = (size_t)m->vals[s->inp[0]].r, S = (size_t)m->vals[s->inp[s->n_in > 1 ? 1 : 0]].r;
    const size_t B = (size_t)q->row / Rq, H = (size_t)s->heads, dk = (size_t)s->dk, dv = (size_t)s->dv;
    if (B == 0) return 0;
    if (kernix_dense(q, s->w, s->tmp[0], s->bias, KERNIX_ACT_NONE, 0.0f) != 0 ||
        kernix_dense(k, s->wk, s->tmp[1], s->bk, KERNIX_ACT_NONE, 0.0f) != 0 ||
        kernix_dense(v, s->wv, s->tmp[2], s->bv, KERNIX_ACT_NONE, 0.0f) != 0) return -1;
    const float* Q = s->tmp[0]->tensor;
    const float* K = s->tmp[1]->tensor;
    const float* V = s->tmp[2]->tensor;
    float* sc = s->tmp[3]->tensor;
    float* ctx = s->tmp[4]->tensor;
    const float scale = 1.0f / sqrtf((float)dk);
    const int causal = s->causal;
    kernix_par(B * H, Rq * S * (dk + dv), ^(size_t i0, size_t ni) {
        for (size_t i = i0; i < i0 + ni; i++) {
            const size_t b = i / H, h = i % H;
            float* p = sc + i * Rq * S;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)Rq, (int)S, (int)dk, scale,
                        Q + b * Rq * H * dk + h * dk, (int)(H * dk), K + b * S * H * dk + h * dk, (int)(H * dk), 0.0f, p, (int)S);
            for (size_t r = 0; r < Rq; r++) {
                float* row = p + r * S;
                if (causal) for (size_t j = r + 1; j < S; j++) row[j] = -INFINITY;
                float mx;
                vDSP_maxv(row, 1, &mx, S);
                const float neg = -mx;
                const int n = (int)S;
                vDSP_vsadd(row, 1, &neg, row, 1, S);
                vvexpf(row, row, &n);
                float sum;
                vDSP_sve(row, 1, &sum, S);
                const float inv = 1.0f / sum;
                vDSP_vsmul(row, 1, &inv, row, 1, S);
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)Rq, (int)dv, (int)S, 1.0f,
                        p, (int)S, V + b * S * H * dv + h * dv, (int)(H * dv), 0.0f, ctx + b * Rq * H * dv + h * dv, (int)(H * dv));
        }
    });
    return kernix_dense(s->tmp[4], s->wo, y, s->bo, KERNIX_ACT_NONE, 0.0f);
}

/* ---- 로드 ---- */
static inline int kernix_read_u32s(kernix_reader* r, uint32_t* v, int n)
{
    for (int i = 0; i < n; i++)
        if (kernix_read_u32(r, &v[i]) != 0) return -1;
    return 0;
}

/* k×n 가중치 블록을 읽음. keep_int8이고 I8_COL이면 *q(int8 묶음)로, 아니면 float로: wt가 있으면 새 float_tensor, 없으면 *wr에 새 배열 */
static inline int kernix_load_weight(kernix_reader* r, int ver, uint64_t k, uint64_t n, int keep_int8,
                                     float_tensor** wt, float** wr, kernix_qweight** q, const char** err)
{
    if (k < 1 || n < 1 || k > INT_MAX || n > INT_MAX || k * n > (uint64_t)INT_MAX) { *err = "가중치 크기가 범위 밖"; return -1; }
    uint32_t dt;
    if (kernix_read_dtype(r, ver >= 2, &dt, err) != 0) return -1;
    if (kernix_weight_bytes(dt, k, n) > r->left) return -1; /* 할당 전에 확인: 거짓 크기로 큰 할당 유도 방지 */
    float* dst = NULL;
    if (!(keep_int8 && dt == KERNIX_DT_I8_COL)) {
        if (wt != NULL) {
            if ((*wt = kernix_float_new((int)k, (int)n, 0)) == NULL) { *err = "메모리 부족"; return -1; }
            dst = (*wt)->tensor;
        } else if ((dst = *wr = malloc(sizeof(float) * k * n)) == NULL) {
            *err = "메모리 부족";
            return -1;
        }
    }
    return kernix_read_weight(r, dt, (int)k, (int)n, keep_int8, dst, q, err);
}

static inline int kernix_geom_ok(uint32_t v, uint32_t lo) { return v >= lo && v <= KERNIX_MAX_GEOM; }

/* 연산 하나의 인자와 가중치를 읽음 (모양 검사는 kernix_infer). ver: 1, 2, 3 */
static inline int kernix_parse_op(kernix_reader* r, int ver, kernix_step* s, const char** err)
{
    uint32_t a[16];
    s->act = KERNIX_ACT_NONE;
    s->dirs = 1;
    s->groups = 1;
    s->kh = s->kw = s->sh = s->sw = s->dh = s->dw = 1;
    switch (s->kind) {
    case KERNIX_OP_LINEAR:
        if (kernix_read_u32s(r, a, 3) != 0) return -1;
        if (a[0] < 1 || a[1] < 1 || a[0] > INT_MAX || a[1] > INT_MAX || a[2] > 1) { *err = "Linear 크기가 범위 밖"; return -1; }
        s->in = (int)a[0];
        s->out = (int)a[1];
        if (kernix_load_weight(r, ver, a[0], a[1], 1, &s->w, NULL, &s->q, err) != 0) return -1;
        return a[2] && (s->bias = kernix_read_alloc(r, a[1])) == NULL ? -1 : 0;
    case KERNIX_OP_ACT:
        if (kernix_read_u32(r, &a[0]) != 0 || (ver >= 3 && kernix_read_floats(r, &s->alpha, 1) != 0)) return -1;
        if (a[0] > (uint32_t)(ver >= 3 ? KERNIX_ACT_LAST : KERNIX_ACT_TANH) || !isfinite(s->alpha)) { *err = "알 수 없는 활성화"; return -1; }
        s->act = (int)a[0];
        return 0;
    case KERNIX_OP_LAYERNORM:
        if (kernix_read_u32s(r, a, 2) != 0 || kernix_read_floats(r, &s->eps, 1) != 0) return -1;
        if (a[0] < 1 || a[0] > INT_MAX || a[1] > 1 || !(s->eps >= 0.0f) || isinf(s->eps)) { *err = "LayerNorm 값이 범위 밖"; return -1; }
        s->in = s->out = (int)a[0];
        return a[1] && ((s->gamma = kernix_read_alloc(r, a[0])) == NULL || (s->beta = kernix_read_alloc(r, a[0])) == NULL) ? -1 : 0;
    case KERNIX_OP_SOFTMAX:
        return 0;
    case KERNIX_OP_LSTM:
    case KERNIX_OP_GRU:
    case KERNIX_OP_RNN: {
        if (ver < 3) { /* KNX1/2: in, units, act, ret_seq, seq. 활성화는 relu/tanh */
            if (kernix_read_u32s(r, a, 5) != 0) return -1;
            if (a[0] < 1 || a[0] > INT_MAX || a[1] < 1 || a[1] > KERNIX_CHUNK / 2 || a[3] > 1 || a[4] < 1 || a[4] > INT_MAX ||
                (a[2] != KERNIX_ACT_RELU && a[2] != KERNIX_ACT_TANH)) { *err = "LSTM 값이 범위 밖"; return -1; }
            s->seq = (int)a[4];
            a[4] = 1;
            a[5] = KERNIX_MERGE_CONCAT;
        } else { /* KNX3: in, units, act, ret_seq, dirs, merge (시퀀스 길이는 입력 모양에서) */
            if (kernix_read_u32s(r, a, 6) != 0) return -1;
            /* units 상한: 게이트 두 개(2·units)를 kernix_act_chunk 한 번으로 처리 */
            if (a[0] < 1 || a[0] > INT_MAX || a[1] < 1 || a[1] > KERNIX_CHUNK / 2 || a[2] > KERNIX_ACT_LAST || a[3] > 1 ||
                a[4] < 1 || a[4] > 2 || a[5] > KERNIX_MERGE_AVE) { *err = "RNN 값이 범위 밖"; return -1; }
        }
        s->in = (int)a[0];
        s->out = (int)a[1];
        s->act = (int)a[2];
        s->ret_seq = (int)a[3];
        s->dirs = (int)a[4];
        s->merge = (int)a[5];
        const uint64_t gu = kernix_gates(s->kind) * (uint64_t)a[1], nb = s->kind == KERNIX_OP_GRU ? 2 * gu : gu;
        for (int d = 0; d < s->dirs; d++) { /* 입력 투영은 항상 float (여러 행이라 AMX가 빠름), 순환 가중치는 int8 유지 가능 */
            if (kernix_load_weight(r, ver, a[0], gu, 0, d ? &s->w_b : &s->w, NULL, NULL, err) != 0 ||
                kernix_load_weight(r, ver, a[1], gu, 1, NULL, d ? &s->rec_b : &s->rec, d ? &s->q_b : &s->q, err) != 0) return -1;
            if ((*(d ? &s->bias_b : &s->bias) = kernix_read_alloc(r, nb)) == NULL) return -1;
        }
        return 0;
    }
    case KERNIX_OP_CONV: { /* cin, cout, kh, kw, sh, sw, dh, dw, pt, pb, pl, pr, groups, has_bias */
        if (kernix_read_u32s(r, a, 14) != 0) return -1;
        if (a[0] < 1 || a[0] > INT_MAX || a[1] < 1 || a[1] > INT_MAX || !kernix_geom_ok(a[2], 1) || !kernix_geom_ok(a[3], 1) ||
            !kernix_geom_ok(a[4], 1) || !kernix_geom_ok(a[5], 1) || !kernix_geom_ok(a[6], 1) || !kernix_geom_ok(a[7], 1) ||
            !kernix_geom_ok(a[8], 0) || !kernix_geom_ok(a[9], 0) || !kernix_geom_ok(a[10], 0) || !kernix_geom_ok(a[11], 0) ||
            a[12] < 1 || a[0] % a[12] != 0 || a[1] % a[12] != 0 || a[13] > 1) { *err = "합성곱 값이 범위 밖"; return -1; }
        s->in = (int)a[0]; s->out = (int)a[1];
        s->kh = (int)a[2]; s->kw = (int)a[3]; s->sh = (int)a[4]; s->sw = (int)a[5]; s->dh = (int)a[6]; s->dw = (int)a[7];
        s->pt = (int)a[8]; s->pb = (int)a[9]; s->pl = (int)a[10]; s->pr = (int)a[11]; s->groups = (int)a[12];
        if (kernix_load_weight(r, ver, (uint64_t)a[2] * a[3] * (a[0] / a[12]), a[1], 0, &s->w, NULL, NULL, err) != 0) return -1;
        return a[13] && (s->bias = kernix_read_alloc(r, a[1])) == NULL ? -1 : 0;
    }
    case KERNIX_OP_POOL: /* type, kh, kw, sh, sw, pt, pb, pl, pr, count_pad */
        if (kernix_read_u32s(r, a, 10) != 0) return -1;
        if (a[0] > KERNIX_POOL_AVG || !kernix_geom_ok(a[1], 1) || !kernix_geom_ok(a[2], 1) || !kernix_geom_ok(a[3], 1) ||
            !kernix_geom_ok(a[4], 1) || a[5] >= a[1] || a[6] >= a[1] || a[7] >= a[2] || a[8] >= a[2] || a[9] > 1) {
            *err = "풀링 값이 범위 밖"; return -1; /* 패딩이 커널 이상이면 빈 창이 생김 */
        }
        s->ptype = (int)a[0]; s->kh = (int)a[1]; s->kw = (int)a[2]; s->sh = (int)a[3]; s->sw = (int)a[4];
        s->pt = (int)a[5]; s->pb = (int)a[6]; s->pl = (int)a[7]; s->pr = (int)a[8]; s->count_pad = (int)a[9];
        return 0;
    case KERNIX_OP_GLOBAL_POOL:
        if (kernix_read_u32(r, &a[0]) != 0) return -1;
        if (a[0] > KERNIX_POOL_AVG) { *err = "풀링 값이 범위 밖"; return -1; }
        s->ptype = (int)a[0];
        return 0;
    case KERNIX_OP_RESHAPE:
    case KERNIX_OP_PERMUTE: {
        if (kernix_read_u32(r, &a[0]) != 0) return -1;
        if (a[0] < 1 || a[0] > KERNIX_MAX_DIMS) { *err = "축 수가 범위 밖"; return -1; }
        s->nd = (int)a[0];
        if (kernix_read_u32s(r, a + 1, s->nd) != 0) return -1;
        int seen = 0;
        for (int k = 0; k < s->nd; k++) {
            const uint32_t v = a[1 + k];
            if (s->kind == KERNIX_OP_RESHAPE ? (v < 1 || v > INT_MAX) : (v >= (uint32_t)s->nd || (seen >> v) & 1)) {
                *err = s->kind == KERNIX_OP_RESHAPE ? "Reshape 모양이 범위 밖" : "Permute 축 순서가 잘못됨"; return -1;
            }
            if (s->kind == KERNIX_OP_RESHAPE) s->d[k] = (int)v;
            else { s->perm[k] = (int)v; seen |= 1 << v; } /* v < nd ≤ 4 */
        }
        return 0;
    }
    case KERNIX_OP_BINARY:
        if (kernix_read_u32(r, &a[0]) != 0) return -1;
        if (a[0] > KERNIX_BIN_MIN) { *err = "알 수 없는 이항 연산"; return -1; }
        s->bop = (int)a[0];
        return 0;
    case KERNIX_OP_CONCAT:
        if (kernix_read_u32(r, &a[0]) != 0) return -1;
        if (a[0] > KERNIX_AXIS_LAST) { *err = "Concatenate 축이 범위 밖"; return -1; }
        s->axis = (int)a[0];
        return 0;
    case KERNIX_OP_EMBEDDING:
        if (kernix_read_u32s(r, a, 2) != 0) return -1;
        s->in = (int)(a[0] > INT_MAX ? 0 : a[0]);
        s->out = (int)(a[1] > INT_MAX ? 0 : a[1]);
        return kernix_load_weight(r, ver, a[0], a[1], 0, &s->w, NULL, NULL, err);
    case KERNIX_OP_AFFINE:
        if (kernix_read_u32(r, &a[0]) != 0) return -1;
        if (a[0] < 1 || a[0] > INT_MAX) { *err = "Affine 크기가 범위 밖"; return -1; }
        s->in = s->out = (int)a[0];
        return (s->gamma = kernix_read_alloc(r, a[0])) == NULL || (s->beta = kernix_read_alloc(r, a[0])) == NULL ? -1 : 0;
    case KERNIX_OP_ADD_CONST:
        if (kernix_read_u32(r, &a[0]) != 0) return -1;
        if (a[0] < 1 || a[0] > INT_MAX) { *err = "상수 크기가 범위 밖"; return -1; }
        s->in = (int)a[0];
        return (s->cst = kernix_read_alloc(r, a[0])) == NULL ? -1 : 0;
    case KERNIX_OP_ATTENTION: { /* heads, dk, dv, causal, cq, cv, ck, cout, has_bias */
        if (kernix_read_u32s(r, a, 9) != 0) return -1;
        if (a[0] < 1 || a[0] > 4096 || a[1] < 1 || a[1] > 65536 || a[2] < 1 || a[2] > 65536 || a[3] > 1 || a[8] > 1 ||
            a[4] < 1 || a[4] > INT_MAX || a[5] < 1 || a[5] > INT_MAX || a[6] < 1 || a[6] > INT_MAX || a[7] < 1 || a[7] > INT_MAX) {
            *err = "어텐션 값이 범위 밖"; return -1;
        }
        s->heads = (int)a[0]; s->dk = (int)a[1]; s->dv = (int)a[2]; s->causal = (int)a[3];
        s->in = (int)a[4]; s->out = (int)a[7];
        const uint64_t hk = (uint64_t)a[0] * a[1], hv = (uint64_t)a[0] * a[2];
        if (kernix_load_weight(r, ver, a[4], hk, 0, &s->w, NULL, NULL, err) != 0 || (a[8] && (s->bias = kernix_read_alloc(r, hk)) == NULL) ||
            kernix_load_weight(r, ver, a[6], hk, 0, &s->wk, NULL, NULL, err) != 0 || (a[8] && (s->bk = kernix_read_alloc(r, hk)) == NULL) ||
            kernix_load_weight(r, ver, a[5], hv, 0, &s->wv, NULL, NULL, err) != 0 || (a[8] && (s->bv = kernix_read_alloc(r, hv)) == NULL) ||
            kernix_load_weight(r, ver, hv, a[7], 0, &s->wo, NULL, NULL, err) != 0 || (a[8] && (s->bo = kernix_read_alloc(r, a[7])) == NULL)) return -1;
        return 0;
    }
    case KERNIX_OP_SELECT: /* axis, start, len, squeeze */
        if (kernix_read_u32s(r, a, 4) != 0) return -1;
        if (a[0] > KERNIX_AXIS_LAST || a[1] > INT_MAX || a[2] < 1 || a[2] > INT_MAX || a[3] > 1) { *err = "Select 값이 범위 밖"; return -1; }
        s->axis = (int)a[0]; s->start = (int)a[1]; s->len = (int)a[2]; s->squeeze = (int)a[3];
        return 0;
    case KERNIX_OP_PAD:
        if (kernix_read_u32s(r, a, 4) != 0) return -1;
        if (!kernix_geom_ok(a[0], 0) || !kernix_geom_ok(a[1], 0) || !kernix_geom_ok(a[2], 0) || !kernix_geom_ok(a[3], 0)) {
            *err = "패딩 값이 범위 밖"; return -1;
        }
        s->pt = (int)a[0]; s->pb = (int)a[1]; s->pl = (int)a[2]; s->pr = (int)a[3];
        return 0;
    case KERNIX_OP_UPSAMPLE:
        if (kernix_read_u32s(r, a, 2) != 0) return -1;
        if (!kernix_geom_ok(a[0], 1) || !kernix_geom_ok(a[1], 1)) { *err = "UpSampling 값이 범위 밖"; return -1; }
        s->sh = (int)a[0]; s->sw = (int)a[1];
        return 0;
    default:
        *err = "알 수 없는 연산";
        return -1;
    }
}

/* 모양 nd, d로 값을 채움. 크기가 int 범위를 넘으면 -1 */
static inline int kernix_val_set(kernix_val* v, int nd, const int* d)
{
    if (nd < 1 || nd > KERNIX_MAX_DIMS) return -1;
    uint64_t r = 1;
    for (int k = 0; k < nd; k++) {
        if (d[k] < 1) return -1;
        if (k < nd - 1 && (r *= (uint64_t)d[k]) > INT_MAX) return -1;
    }
    if (r * (uint64_t)d[nd - 1] > INT_MAX) return -1;
    memset(v, 0, sizeof *v);
    v->nd = nd;
    memcpy(v->d, d, sizeof(int) * (size_t)nd);
    v->r = (int)r;
    v->c = d[nd - 1];
    v->st = -1;
    v->last = -1;
    return 0;
}

/* 단계 s의 출력 모양을 입력 모양에서 계산하고 검사 */
static inline int kernix_infer(kernix_step* s, int ver, const kernix_val* fv, kernix_val* out, const char** err)
{
    const kernix_val* x = &fv[s->inp[0]];
    int nd = x->nd, d[KERNIX_MAX_DIMS];
    memcpy(d, x->d, sizeof d);
    switch (s->kind) {
    case KERNIX_OP_LINEAR:
        if (x->c != s->in) { *err = "Linear 입력 크기가 앞 단계 출력과 다름"; return -1; }
        d[nd - 1] = s->out;
        break;
    case KERNIX_OP_ACT:
    case KERNIX_OP_SOFTMAX:
        break;
    case KERNIX_OP_LAYERNORM:
        if (x->c != s->in) { *err = "LayerNorm 크기가 앞 단계 출력과 다름"; return -1; }
        break;
    case KERNIX_OP_LSTM:
    case KERNIX_OP_GRU:
    case KERNIX_OP_RNN: {
        if (x->c != s->in) { *err = s->kind == KERNIX_OP_LSTM ? "LSTM 입력 크기가 앞 단계 출력과 다름" : "RNN 입력 크기가 앞 단계 출력과 다름"; return -1; }
        if (ver < 3 && s->seq != x->r) { *err = "LSTM 시퀀스 길이가 앞 단계와 다름"; return -1; }
        s->seq = x->r;
        const int uo = s->dirs == 2 && s->merge == KERNIX_MERGE_CONCAT ? 2 * s->out : s->out;
        if (s->ret_seq) { nd = 2; d[0] = x->r; d[1] = uo; }
        else { nd = 1; d[0] = uo; }
        break;
    }
    case KERNIX_OP_CONV:
    case KERNIX_OP_POOL:
    case KERNIX_OP_PAD:
    case KERNIX_OP_UPSAMPLE: {
        if (nd != 2 && nd != 3) { *err = "합성곱·풀링 입력은 T×C 또는 H×W×C여야 함"; return -1; }
        if (nd == 2 && (s->kw != 1 || s->sw != 1 || s->dw != 1 || s->pl != 0 || s->pr != 0)) {
            *err = "1차원 입력(T×C)에 2차원 커널"; return -1;
        }
        if (s->kind == KERNIX_OP_CONV && x->c != s->in) { *err = "합성곱 입력 채널이 앞 단계 출력과 다름"; return -1; }
        const int64_t H = d[0], W = nd == 3 ? d[1] : 1;
        int64_t Ho, Wo;
        if (s->kind == KERNIX_OP_PAD) { Ho = H + s->pt + s->pb; Wo = W + s->pl + s->pr; }
        else if (s->kind == KERNIX_OP_UPSAMPLE) { Ho = H * s->sh; Wo = W * s->sw; }
        else {
            const int64_t nh = H + s->pt + s->pb - ((int64_t)s->dh * (s->kh - 1) + 1), nw = W + s->pl + s->pr - ((int64_t)s->dw * (s->kw - 1) + 1);
            if (nh < 0 || nw < 0) { *err = "합성곱·풀링 출력 크기가 0 이하"; return -1; }
            Ho = nh / s->sh + 1;
            Wo = nw / s->sw + 1;
        }
        if (Ho > INT_MAX || Wo > INT_MAX) { *err = "모양이 범위 밖"; return -1; }
        d[0] = (int)Ho;
        if (nd == 3) d[1] = (int)Wo;
        if (s->kind == KERNIX_OP_CONV) d[nd - 1] = s->out;
        break;
    }
    case KERNIX_OP_GLOBAL_POOL:
        nd = 1;
        d[0] = x->c;
        break;
    case KERNIX_OP_RESHAPE: {
        uint64_t n = 1;
        for (int k = 0; k < s->nd; k++) n *= (uint64_t)s->d[k];
        if (n != (uint64_t)x->r * (uint64_t)x->c) { *err = "Reshape 크기가 입력과 다름"; return -1; }
        nd = s->nd;
        memcpy(d, s->d, sizeof d);
        break;
    }
    case KERNIX_OP_PERMUTE:
        if (s->nd != nd) { *err = "Permute 축 수가 입력과 다름"; return -1; }
        for (int k = 0; k < nd; k++) d[k] = x->d[s->perm[k]];
        break;
    case KERNIX_OP_BINARY: {
        const kernix_val* y = &fv[s->inp[1]];
        const int R = x->r > y->r ? x->r : y->r, C = x->c > y->c ? x->c : y->c;
        const kernix_val* big = (int64_t)x->r * x->c >= (int64_t)y->r * y->c ? x : y;
        if ((x->r != R && x->r != 1) || (y->r != R && y->r != 1) || (x->c != C && x->c != 1) || (y->c != C && y->c != 1) || big->r != R) {
            *err = "이항 연산 입력 모양이 맞지 않음"; return -1;
        }
        nd = big->nd;
        memcpy(d, big->d, sizeof d);
        d[nd - 1] = C;
        break;
    }
    case KERNIX_OP_CONCAT: {
        int64_t sum = 0;
        for (int k = 0; k < s->n_in; k++) {
            const kernix_val* v = &fv[s->inp[k]];
            int same = v->nd == nd;
            for (int j = 0; same && j < nd; j++)
                if (j != (s->axis == KERNIX_AXIS_FIRST ? 0 : nd - 1)) same = v->d[j] == x->d[j];
            if (!same) { *err = "Concatenate 입력 모양이 맞지 않음"; return -1; }
            sum += v->d[s->axis == KERNIX_AXIS_FIRST ? 0 : nd - 1];
        }
        if (sum > INT_MAX) { *err = "모양이 범위 밖"; return -1; }
        d[s->axis == KERNIX_AXIS_FIRST ? 0 : nd - 1] = (int)sum;
        break;
    }
    case KERNIX_OP_EMBEDDING:
        if (nd >= KERNIX_MAX_DIMS) { *err = "축 수가 범위 밖"; return -1; }
        d[nd++] = s->out;
        break;
    case KERNIX_OP_AFFINE:
        if (x->c != s->in) { *err = "Affine 크기가 앞 단계 출력과 다름"; return -1; }
        break;
    case KERNIX_OP_ADD_CONST:
        if ((int64_t)x->r * x->c != s->in) { *err = "상수 크기가 앞 단계 출력과 다름"; return -1; }
        break;
    case KERNIX_OP_ATTENTION: {
        const kernix_val* v = &fv[s->inp[s->n_in > 1 ? 1 : 0]];
        const kernix_val* k = &fv[s->inp[s->n_in > 2 ? 2 : s->n_in > 1 ? 1 : 0]];
        if (x->c != s->w->row || v->c != s->wv->row || k->c != s->wk->row || k->r != v->r) {
            *err = "어텐션 입력 크기가 가중치와 다름"; return -1;
        }
        d[nd - 1] = s->out;
        break;
    }
    case KERNIX_OP_SELECT: {
        const int ax = s->axis == KERNIX_AXIS_FIRST ? 0 : nd - 1;
        if ((int64_t)s->start + s->len > d[ax]) { *err = "Select 범위가 입력 밖"; return -1; }
        d[ax] = s->len;
        if (s->squeeze && s->len == 1 && nd > 1) {
            memmove(d + ax, d + ax + 1, sizeof(int) * (size_t)(nd - 1 - ax));
            nd--;
        }
        break;
    }
    default:
        break;
    }
    if (kernix_val_set(out, nd, d) != 0) { *err = "모양이 범위 밖"; return -1; }
    return 0;
}

/* 단계 i의 출력을 입력 저장 버퍼 위에서 계산해도 되는 연산 (원소별·행별로 같은 위치만 읽고 씀) */
static inline int kernix_inplace_ok(int kind)
{
    return kind == KERNIX_OP_ACT || kind == KERNIX_OP_LAYERNORM || kind == KERNIX_OP_SOFTMAX ||
           kind == KERNIX_OP_AFFINE || kind == KERNIX_OP_ADD_CONST || kind == KERNIX_OP_BINARY;
}

/* 값마다 저장 버퍼를 정함. 수명이 끝난 버퍼는 다음 값이 다시 쓰고, Reshape는 입력 버퍼를 그대로 보며,
 * 입력을 마지막으로 읽는 제자리 가능 연산은 입력 버퍼에 그대로 씀 */
static inline int kernix_plan(kernix_model* m)
{
    const int n = m->n_steps;
    size_t* cap = calloc((size_t)n + 1, sizeof(size_t));
    int* busy = calloc((size_t)n + 1, sizeof(int)); /* 저장 버퍼마다 그 안의 값들을 마지막으로 읽는 단계 */
    if (cap == NULL || busy == NULL) { free(cap); free(busy); return -1; }
    int ns = 0;
    for (int i = 0; i < n; i++) {
        const kernix_step* s = &m->steps[i];
        kernix_val* v = &m->vals[i + 1];
        const size_t need = (size_t)v->r * (size_t)v->c;
        int st = -2;
        if (s->kind == KERNIX_OP_RESHAPE) {
            st = m->vals[s->inp[0]].st; /* -1(호출자 입력)의 보기도 그대로 */
        } else if (kernix_inplace_ok(s->kind)) {
            for (int k = 0; k < (s->kind == KERNIX_OP_BINARY ? 2 : 1) && st == -2; k++) {
                const kernix_val* u = &m->vals[s->inp[k]];
                if (u->st >= 0 && busy[u->st] <= i && (size_t)u->r * (size_t)u->c == need) st = u->st;
            }
        }
        if (st == -2) { /* 비어 있는 버퍼 중 충분히 크면서 가장 작은 것, 없으면 가장 큰 것을 늘림 */
            int best = -1;
            for (int k = 0; k < ns; k++) {
                if (busy[k] >= i) continue;
                if (best < 0) best = k;
                else if (cap[k] >= need ? (cap[best] < need || cap[k] < cap[best]) : (cap[best] < need && cap[k] > cap[best])) best = k;
            }
            if (best < 0) best = ns++;
            if (cap[best] < need) cap[best] = need;
            st = best;
        }
        v->st = st;
        if (st >= 0 && busy[st] < v->last) busy[st] = v->last;
        if (st >= 0 && busy[st] < i) busy[st] = i; /* 출력을 쓰는 동안 사용 중 */
    }
    free(busy);
    m->n_st = ns;
    m->st_cap = cap;
    m->st = calloc((size_t)(ns ? ns : 1), sizeof(float_tensor*));
    return m->st == NULL ? -1 : 0;
}

/* 실패 시 NULL (이유는 stderr) */
static inline kernix_model* kernix_model_load(const char* path)
{
    FILE* f = path ? fopen(path, "rb") : NULL;
    if (f == NULL) {
        fprintf(stderr, "모델 로드 실패 (%s): 파일을 열 수 없음\n", path ? path : "NULL");
        return NULL;
    }
    kernix_model* m = NULL;
    kernix_val* fv = NULL;
    int* cons = NULL;
    int* map = NULL;
    const char* err = "파일이 선언된 크기보다 짧음";
    kernix_reader r = { f, 0 };
    long size;
    uint32_t magic, n_ops, hdr[2 + KERNIX_MAX_DIMS];
    if (fseek(f, 0, SEEK_END) != 0 || (size = ftell(f)) < 0 || fseek(f, 0, SEEK_SET) != 0) goto fail;
    r.left = (size_t)size;

    if (kernix_read_u32(&r, &magic) != 0 || kernix_read_u32(&r, &n_ops) != 0) goto fail;
    const int ver = memcmp(&magic, "KNX1", 4) == 0 ? 1 : memcmp(&magic, "KNX2", 4) == 0 ? 2 : memcmp(&magic, "KNX3", 4) == 0 ? 3 : 0;
    if (ver == 0) { err = "KNX1/KNX2/KNX3 형식이 아님"; goto fail; }
    if (n_ops < 1 || n_ops > KERNIX_MODEL_MAX_OPS) { err = "연산 수가 범위 밖"; goto fail; }

    m = calloc(1, sizeof *m);
    if (m != NULL) m->steps = calloc(n_ops, sizeof(kernix_step));
    fv = calloc((size_t)n_ops + 1, sizeof(kernix_val));
    cons = calloc((size_t)n_ops + 1, sizeof(int));
    map = calloc((size_t)n_ops + 1, sizeof(int));
    if (m == NULL || m->steps == NULL || fv == NULL || cons == NULL || map == NULL) { err = "메모리 부족"; goto fail; }
    m->batch = -1;

    if (ver == 3) { /* layout, 입력 축 수, 표본 하나의 입력 모양 (channels-last) */
        if (kernix_read_u32s(&r, hdr, 2) != 0) goto fail;
        if (hdr[0] > 1 || hdr[1] < 1 || hdr[1] > KERNIX_MAX_DIMS) { err = "입력 모양이 범위 밖"; goto fail; }
        int d[KERNIX_MAX_DIMS];
        if (kernix_read_u32s(&r, hdr + 2, (int)hdr[1]) != 0) goto fail;
        for (uint32_t k = 0; k < hdr[1]; k++) d[k] = hdr[2 + k] > INT_MAX ? 0 : (int)hdr[2 + k];
        if (kernix_val_set(&fv[0], (int)hdr[1], d) != 0) { err = "입력 모양이 범위 밖"; goto fail; }
        m->layout = (int)hdr[0];
    }
    for (uint32_t i = 0; i < n_ops; i++) {
        kernix_step* s = &m->steps[i];
        uint32_t kind;
        if (kernix_read_u32(&r, &kind) != 0) goto fail;
        if (kind > (uint32_t)(ver == 3 ? KERNIX_OP_LAST : KERNIX_OP_LSTM)) { err = "알 수 없는 연산"; goto fail; }
        s->kind = (int)kind;
        if (ver == 3) { /* 입력 값 번호: 0 = 모델 입력, k+1 = 연산 k의 출력 (앞 연산만) */
            uint32_t n_in;
            if (kernix_read_u32(&r, &n_in) != 0) goto fail;
            const uint32_t lo = kind == KERNIX_OP_BINARY ? 2 : kind == KERNIX_OP_CONCAT ? 2 : 1;
            const uint32_t hi = kind == KERNIX_OP_BINARY ? 2 : kind == KERNIX_OP_CONCAT ? KERNIX_MAX_INPUTS : kind == KERNIX_OP_ATTENTION ? 3 : 1;
            if (n_in < lo || n_in > hi) { err = "연산 입력 수가 범위 밖"; goto fail; }
            s->n_in = (int)n_in;
            for (uint32_t k = 0; k < n_in; k++) {
                uint32_t id;
                if (kernix_read_u32(&r, &id) != 0) goto fail;
                if (id > i) { err = "연산 입력이 뒤 연산을 가리킴"; goto fail; }
                s->inp[k] = (int)id;
            }
        } else {
            s->n_in = 1;
            s->inp[0] = (int)i;
        }
        m->n_steps = (int)i + 1; /* 이후 실패해도 kernix_model_free가 이 단계까지 해제 */
        if (kernix_parse_op(&r, ver, s, &err) != 0) goto fail;
    }
    if (r.left != 0) { err = "마지막 연산 뒤에 데이터가 남음"; goto fail; }

    if (ver < 3) { /* 입력 모양: 첫 크기 있는 연산의 입력 크기, LSTM이 있으면 첫 LSTM의 시퀀스 길이 × 그 크기 */
        int in_dim = 0, seq = 0;
        for (uint32_t i = 0; i < n_ops; i++) {
            const kernix_step* s = &m->steps[i];
            if (in_dim == 0 && (s->kind == KERNIX_OP_LINEAR || s->kind == KERNIX_OP_LAYERNORM || s->kind == KERNIX_OP_LSTM)) in_dim = s->in;
            if (seq == 0 && s->kind == KERNIX_OP_LSTM) seq = s->seq;
        }
        if (in_dim == 0) { err = "Linear나 LayerNorm이 없어 입력 크기를 알 수 없음"; goto fail; }
        const int d[2] = { seq ? seq : in_dim, in_dim };
        if (kernix_val_set(&fv[0], seq ? 2 : 1, d) != 0) { err = "입력 모양이 범위 밖"; goto fail; }
    }
    for (uint32_t i = 0; i < n_ops; i++)
        if (kernix_infer(&m->steps[i], ver, fv, &fv[i + 1], &err) != 0) goto fail;

    /* 행렬곱·합성곱 출력을 그 활성화만 읽으면 활성화를 합침 (결과가 캐시에 있을 때 한 번에, GPU 행은 GPU에서) */
    for (uint32_t i = 0; i < n_ops; i++)
        for (int k = 0; k < m->steps[i].n_in; k++) cons[m->steps[i].inp[k]]++;
    cons[n_ops]++; /* 모델 출력 */
    int n = 0;
    for (uint32_t i = 0; i < n_ops; i++) {
        kernix_step s = m->steps[i];
        for (int k = 0; k < s.n_in; k++) s.inp[k] = map[s.inp[k]];
        if (s.kind == KERNIX_OP_ACT && s.inp[0] > 0 && cons[m->steps[i].inp[0]] == 1) {
            kernix_step* p = &m->steps[s.inp[0] - 1]; /* 내부 번호: 이미 앞쪽으로 옮겨졌음 */
            if ((p->kind == KERNIX_OP_LINEAR || p->kind == KERNIX_OP_CONV) && p->act == KERNIX_ACT_NONE) {
                p->act = s.act;
                p->alpha = s.alpha;
                map[i + 1] = s.inp[0];
                continue;
            }
        }
        m->steps[n] = s;
        fv[n + 1] = fv[i + 1];
        map[i + 1] = n + 1;
        n++;
    }
    if (n < (int)n_ops) memset(&m->steps[n], 0, sizeof(kernix_step) * (n_ops - (uint32_t)n)); /* 옮겨진 단계의 중복 사본 */
    m->n_steps = n;
    m->n_vals = n + 1;
    m->out_v = map[n_ops];
    m->vals = fv;
    fv = NULL;
    m->views = calloc((size_t)m->n_vals, sizeof(float_tensor));
    if (m->views == NULL) { err = "메모리 부족"; goto fail; }
    for (int i = 0; i < n; i++)
        for (int k = 0; k < m->steps[i].n_in; k++) m->vals[m->steps[i].inp[k]].last = i;
    m->vals[m->out_v].last = n;
    if (kernix_plan(m) != 0) { err = "메모리 부족"; goto fail; }

    const kernix_val* vin = &m->vals[0];
    const kernix_val* vout = &m->vals[m->out_v];
    m->in_dim = vin->c;
    m->seq_len = vin->r;
    m->out_dim = vout->r * vout->c;
    m->in_nd = vin->nd;
    memcpy(m->in_shape, vin->d, sizeof m->in_shape);
    m->out_nd = vout->nd;
    memcpy(m->out_shape, vout->d, sizeof m->out_shape);
    m->ws_limit = KERNIX_WORKSPACE_LIMIT;
    if ((m->ws_per_sample = kernix_model_shape(m, m->seq_len, KERNIX_SHAPE_COUNT)) == KERNIX_SHAPE_FAIL) {
        err = "작업 버퍼 크기가 범위 밖"; goto fail;
    }
    for (int i = 0; i < m->n_steps; i++) {
        if (m->steps[i].q != NULL) m->ws_fixed += (size_t)m->steps[i].q->k * (size_t)m->steps[i].q->n;
        if (m->steps[i].q_b != NULL) m->ws_fixed += (size_t)m->steps[i].q_b->k * (size_t)m->steps[i].q_b->n;
    }
    free(cons);
    free(map);
    fclose(f);
    return m;

fail:
    fprintf(stderr, "모델 로드 실패 (%s): %s\n", path, err);
    kernix_model_free(m);
    free(fv);
    free(cons);
    free(map);
    fclose(f);
    return NULL;
}

/* ---- 작업 버퍼 · 실행 ---- */
/* 작업 버퍼 하나: COUNT는 크기만 셈, ALLOC은 새로 할당, VIEW는 이미 할당된 (더 큰) 버퍼의 행 수만 rows로 맞춤 */
static inline int kernix_ws(float_tensor** t, size_t rows, size_t cols, int mode, size_t* n)
{
    if (rows > INT_MAX || cols > INT_MAX) return -1;
    *n += rows * cols;
    if (mode == KERNIX_SHAPE_VIEW) (*t)->row = (int)rows;
    else if (mode == KERNIX_SHAPE_ALLOC && (*t = kernix_float_new((int)rows, (int)cols, 0)) == NULL) return -1;
    return 0;
}

/* int8 가중치의 float 사본: 그 단계의 행 수가 임계값을 넘을 때만 ALLOC에서 할당하고 풀어 둠.
 * VIEW는 그대로(가중치라 행 수와 무관), COUNT는 세지 않음 (ws_fixed로 따로 계산) */
static inline int kernix_ws_deq(const kernix_qweight* q, float_tensor** deq, size_t rows, int mode)
{
    if (q == NULL || rows <= KERNIX_INT8_MAX_ROWS || mode != KERNIX_SHAPE_ALLOC) return 0;
    if ((*deq = kernix_float_new(q->k, q->n, 0)) == NULL) return -1;
    kernix_qweight_unpack(q, (*deq)->tensor);
    return 0;
}

/* 입력 rows행일 때의 모든 작업 버퍼에 mode를 적용. 반환: 작업 버퍼 float 수 (0행이면 0), 실패 시 KERNIX_SHAPE_FAIL */
static inline size_t kernix_model_shape(kernix_model* m, int rows, int mode)
{
    size_t n = 0;
    const size_t B = (size_t)rows / (size_t)m->seq_len;
    for (int k = 0; k < m->n_st; k++)
        if (kernix_ws(&m->st[k], B, m->st_cap[k], mode, &n) != 0) return KERNIX_SHAPE_FAIL;
    for (int i = 0; i < m->n_steps; i++) {
        kernix_step* s = &m->steps[i];
        const kernix_val* vi = &m->vals[s->inp[0]];
        const kernix_val* vo = &m->vals[i + 1];
        const size_t rin = B * (size_t)vi->r, rout = B * (size_t)vo->r;
        if (mode != KERNIX_SHAPE_COUNT) s->buf = &m->views[i + 1];
        switch (s->kind) {
        case KERNIX_OP_LINEAR:
            if (kernix_ws_deq(s->q, &s->deq, rin, mode) != 0) return KERNIX_SHAPE_FAIL;
            break;
        case KERNIX_OP_LSTM:
        case KERNIX_OP_GRU:
        case KERNIX_OP_RNN: {
            const size_t u = (size_t)s->out, gu = kernix_gates(s->kind) * u;
            if (kernix_ws_deq(s->q, &s->deq, B, mode) != 0 || kernix_ws_deq(s->q_b, &s->deq_b, B, mode) != 0 ||
                kernix_ws(&s->tmp[0], rin, gu, mode, &n) != 0 || kernix_ws(&s->tmp[1], B, gu, mode, &n) != 0 ||
                (s->kind == KERNIX_OP_LSTM && kernix_ws(&s->tmp[2], B, u, mode, &n) != 0) ||
                kernix_ws(&s->tmp[3], B, u, mode, &n) != 0 ||
                (s->kind == KERNIX_OP_GRU && kernix_ws(&s->tmp[4], B, gu, mode, &n) != 0) ||
                (s->dirs == 2 && s->merge != KERNIX_MERGE_CONCAT && kernix_ws(&s->tmp[5], s->ret_seq ? rin : B, u, mode, &n) != 0))
                return KERNIX_SHAPE_FAIL;
            break;
        }
        case KERNIX_OP_CONV: {
            const int cg = s->in / s->groups;
            if (((s->groups == 1 && !kernix_conv_pointwise(s)) || (s->groups > 1 && cg > 1)) &&
                kernix_ws(&s->tmp[0], rout, (size_t)s->kh * s->kw * cg, mode, &n) != 0) return KERNIX_SHAPE_FAIL;
            break;
        }
        case KERNIX_OP_ATTENTION: {
            const size_t S = B * (size_t)m->vals[s->inp[s->n_in > 1 ? 1 : 0]].r, H = (size_t)s->heads;
            if (kernix_ws(&s->tmp[0], rin, H * s->dk, mode, &n) != 0 || kernix_ws(&s->tmp[1], S, H * s->dk, mode, &n) != 0 ||
                kernix_ws(&s->tmp[2], S, H * s->dv, mode, &n) != 0 ||
                kernix_ws(&s->tmp[3], rin * H, S / (B ? B : 1), mode, &n) != 0 ||
                kernix_ws(&s->tmp[4], rin, H * s->dv, mode, &n) != 0) return KERNIX_SHAPE_FAIL;
            break;
        }
        default:
            break;
        }
    }
    return n;
}

/* 입력 행 수 rows에 맞춰 작업 버퍼를 다시 할당. 새로 잡기 전에 기존 것을 먼저 해제해 최대 메모리를 낮춤.
 * 실패 시 -1 (batch는 -1로 남아 다음 호출이 다시 시도) */
static inline int kernix_model_alloc(kernix_model* m, int rows)
{
    float_tensor* out_all = m->out_all; /* 전체 출력은 조각 크기와 무관하므로 유지 */
    m->out_all = NULL;
    kernix_model_release(m); /* 재할당: Metal 캐시는 유지해 빠르게 */
    m->out_all = out_all;
    if (kernix_model_shape(m, rows, KERNIX_SHAPE_ALLOC) == KERNIX_SHAPE_FAIL) return -1;
    m->batch = rows;
    return 0;
}

/* 작업 버퍼 상한 (바이트). 0이면 제한 없이 배치 전체를 한 번에 실행 (가장 빠를 수 있지만 메모리가 배치에 비례) */
static inline void kernix_model_set_workspace_limit(kernix_model* m, size_t bytes)
{
    if (m != NULL) m->ws_limit = bytes;
}

/* 작업 버퍼가 x->row행에 맞춰져 있을 때 모든 단계를 한 번 실행 */
static inline const float_tensor* kernix_model_run(kernix_model* m, const float_tensor* x)
{
    const int B = x->row / m->seq_len;
    for (int v = 0; v < m->n_vals; v++) {
        const kernix_val* val = &m->vals[v];
        m->views[v] = (float_tensor){ B * val->r, val->c, val->st < 0 ? x->tensor : m->st[val->st]->tensor };
    }
    m->out_view = (float_tensor){ B, m->out_dim, m->views[m->out_v].tensor };
    if (B == 0) return &m->out_view;
    for (int i = 0; i < m->n_steps; i++) {
        kernix_step* s = &m->steps[i];
        const float_tensor* a = &m->views[s->inp[0]];
        float_tensor* y = &m->views[i + 1];
        const kernix_val* vi = &m->vals[s->inp[0]];
        const kernix_val* vo = &m->vals[i + 1];
        const size_t rows = (size_t)y->row, cols = (size_t)y->col;
        switch (s->kind) {
        case KERNIX_OP_LINEAR:
            if (s->q != NULL && a->row <= KERNIX_INT8_MAX_ROWS) { /* 행이 적은 int8: sdot (bias는 커널이 더함) */
                if (kernix_qmatmul(a->tensor, a->row, s->q, s->bias, 0, y->tensor) != 0) return NULL;
                kernix_bias_act(y->tensor, rows, cols, NULL, s->act, s->alpha);
            } else if (kernix_dense(a, s->q != NULL ? s->deq : s->w, y, s->bias, s->act, s->alpha) != 0) { /* 사본이 없으면 -1 */
                return NULL;
            }
            break;
        case KERNIX_OP_LSTM:
        case KERNIX_OP_GRU:
        case KERNIX_OP_RNN:
            if (kernix_rnn_run(s, a, y) != 0) return NULL;
            break;
        case KERNIX_OP_ACT:
            kernix_copy_if(a, y);
            kernix_bias_act(y->tensor, rows, cols, NULL, s->act, s->alpha);
            break;
        case KERNIX_OP_LAYERNORM: kernix_layer_norm_rows_into(a->tensor, y->tensor, rows, cols, s->gamma, s->beta, s->eps); break;
        case KERNIX_OP_SOFTMAX: floats_softmax_into(a->tensor, y->tensor, rows, cols); break;
        case KERNIX_OP_CONV:
            if (kernix_conv_run(s, vi, a, vo, y) != 0) return NULL;
            break;
        case KERNIX_OP_POOL: kernix_pooling_run(s, vi, a, vo, y); break;
        case KERNIX_OP_GLOBAL_POOL: kernix_global_pool_run(s, vi, a, y); break;
        case KERNIX_OP_RESHAPE: break; /* 같은 버퍼의 보기 */
        case KERNIX_OP_PERMUTE: kernix_permute_run(s, vi, a, y); break;
        case KERNIX_OP_BINARY: kernix_binary_run(s, vi, a, &m->vals[s->inp[1]], &m->views[s->inp[1]], vo, y); break;
        case KERNIX_OP_CONCAT: kernix_concat_run(s, m, vo, y); break;
        case KERNIX_OP_EMBEDDING: kernix_embedding_run(s, a, y); break;
        case KERNIX_OP_AFFINE: {
            const float* g = s->gamma;
            const float* b = s->beta;
            const float* xin = a->tensor;
            float* o = y->tensor;
            kernix_par(rows, cols, ^(size_t r0, size_t nr) {
                for (size_t r = r0; r < r0 + nr; r++) vDSP_vma(xin + r * cols, 1, g, 1, b, 1, o + r * cols, 1, cols);
            });
            break;
        }
        case KERNIX_OP_ADD_CONST: {
            const size_t per = (size_t)s->in;
            for (int b = 0; b < B; b++) vDSP_vadd(a->tensor + b * per, 1, s->cst, 1, y->tensor + b * per, 1, per);
            break;
        }
        case KERNIX_OP_ATTENTION:
            if (kernix_attention_run(s, m, y) != 0) return NULL;
            break;
        case KERNIX_OP_SELECT: kernix_select_run(s, vi, a, y); break;
        case KERNIX_OP_PAD: kernix_pad_run(s, vi, a, vo, y); break;
        case KERNIX_OP_UPSAMPLE: kernix_upsample_run(s, vi, a, vo, y); break;
        }
    }
    return &m->out_view;
}

/* x: (배치·seq_len)×in_dim. 표본마다 seq_len행을 연속으로 둔다 (numpy의 reshape(-1, in_dim), 이미지는 NHWC).
 * 반환: 배치×out_dim (표본마다 출력 모양 out_shape을 펼친 것), 실패 시 NULL.
 * 반환된 텐서는 모델 소유: 해제하지 말 것, 다음 forward가 덮어씀.
 * 작업 버퍼가 ws_limit을 넘을 배치는 표본을 나눠 실행한다 (결과는 같고 최대 메모리만 줄어듦).
 * 한 모델의 forward를 여러 스레드에서 동시에 부르면 안 됨 (버퍼 공유) */
static inline const float_tensor* kernix_model_forward(kernix_model* m, const float_tensor* x)
{
    if (m == NULL || x == NULL || x->tensor == NULL || x->row < 0 || x->col != m->in_dim ||
        x->row % m->seq_len != 0) return NULL;

    /* 모델 자신의 버퍼가 입력으로 오면 연산의 입력과 출력이 겹침 */
    int own = m->out_all != NULL && x->tensor == m->out_all->tensor;
    for (int k = 0; k < m->n_st; k++) own |= m->st[k] != NULL && x->tensor == m->st[k]->tensor;
    if (own) {
        fprintf(stderr, "forward: 이 모델의 출력 버퍼는 같은 모델의 입력으로 쓸 수 없음 (복사해서 전달)\n");
        return NULL;
    }

    const int samples = x->row / m->seq_len;
    int chunk = samples; /* 한 번에 실행할 표본 수 */
    if (m->ws_limit > 0 && m->ws_per_sample > 0) { /* int8 사본(ws_fixed)은 조각 크기와 무관한 고정분이라 먼저 뺌 */
        const size_t fixed = m->ws_fixed * sizeof(float), avail = m->ws_limit > fixed ? m->ws_limit - fixed : 0;
        size_t c = avail / (m->ws_per_sample * sizeof(float));
        if (c < 1) c = 1;
        if (c < (size_t)samples) chunk = (int)c;
    }
    const int crow = chunk * m->seq_len;
    if (crow != m->batch && kernix_model_alloc(m, crow) != 0) return NULL; /* 조각 크기가 바뀔 때만 할당 */

    if (chunk == samples) {
        kernix_model_shape(m, crow, KERNIX_SHAPE_VIEW); /* 이전 호출의 마지막 조각이 행 수를 줄여 놨을 수 있음 */
        return kernix_model_run(m, x);
    }
    if (m->out_all == NULL || m->out_all->row != samples) {
        float_free(m->out_all);
        if ((m->out_all = kernix_float_new(samples, m->out_dim, 0)) == NULL) return NULL;
    }
    for (int s0 = 0; s0 < samples; s0 += chunk) {
        const int n = samples - s0 < chunk ? samples - s0 : chunk; /* 마지막 조각은 같은 버퍼를 행 수만 줄여 씀 */
        const float_tensor part = { n * m->seq_len, x->col, x->tensor + (size_t)s0 * (size_t)m->seq_len * (size_t)x->col };
        kernix_model_shape(m, part.row, KERNIX_SHAPE_VIEW);
        const float_tensor* y = kernix_model_run(m, &part);
        if (y == NULL) return NULL;
        memcpy(m->out_all->tensor + (size_t)s0 * (size_t)m->out_dim, y->tensor, sizeof(float) * (size_t)n * (size_t)m->out_dim);
    }
    return m->out_all;
}

#endif
