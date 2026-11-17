#include "kernix_model.h"
#include <time.h>
#include <unistd.h>
#include <sys/resource.h>
#include <mach/mach.h>

/* MLP 런타임 정확성 검증 + 벤치마크. 데이터는 test/mlp_bench.py가 test/model_data/에 생성.
 *   --check            검증만
 *   --bench runtime    런타임 forward 측정
 *   --bench naive      기존 연산을 이어 붙인 방식(런타임 도입 전) 측정
 *   --eval <이름> [<데이터>]  외부 모델 평가 (test/sign_bench.py가 호출). 데이터 기본값: 이름에서 _int8을 뺀 것
 *   --qbench           int8 커널 vs Accelerate 행 수별 시간 (KERNIX_INT8_MAX_ROWS 측정)
 *   --mem <이름> <MB>  작업 버퍼 상한(MB, 0 = 제한 없음)별 전체 배치 시간·메모리, trim 효과 (test/memory_bench.py가 호출) */

#define DIR "test/model_data/"
#define MAGIC 0x31584E4Bu /* "KNX1" 리틀 엔디언 */
#define ONE 0x3F800000u   /* 1.0f 비트 */

static int fail;
#define CHECK(c) do { if (!(c)) { printf("  ✗ %s:%d  %s\n", __FILE__, __LINE__, #c); fail = 1; } } while (0)

static void write_file(const char* path, const void* data, size_t n)
{
    FILE* f = fopen(path, "wb");
    if (f == NULL || fwrite(data, 1, n, f) != n) { fprintf(stderr, "쓰기 실패: %s\n", path); exit(1); }
    fclose(f);
}

static unsigned char* read_file(const char* path, size_t* n)
{
    FILE* f = fopen(path, "rb");
    if (f == NULL) { fprintf(stderr, "%s 없음: 먼저 python test/mlp_bench.py 실행\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    *n = (size_t)ftell(f);
    rewind(f);
    unsigned char* p = malloc(*n + 4);
    if (p == NULL || fread(p, 1, *n, f) != *n) { fprintf(stderr, "읽기 실패: %s\n", path); exit(1); }
    fclose(f);
    return p;
}

/* 주어진 바이트를 bad.knx로 쓰고, 로드가 거부되면 1 */
static int rejected(const void* data, size_t n)
{
    write_file(DIR "bad.knx", data, n);
    kernix_model* m = kernix_model_load(DIR "bad.knx");
    kernix_model_free(m);
    return m == NULL;
}

static void check_load(void)
{
    kernix_model* m = kernix_model_load(DIR "small.knx");
    CHECK(m != NULL && m->in_dim == 64 && m->out_dim == 10);
    CHECK(m != NULL && m->n_steps == 9); /* 연산 14개 중 Linear 바로 뒤 활성화 5개가 합쳐짐 */
    kernix_model_free(m);
    kernix_model_free(NULL);

    fprintf(stderr, "(아래 '모델 로드 실패' 메시지는 불량 파일 거부 테스트의 정상 출력)\n");
    CHECK(kernix_model_load(DIR "no_such_file.knx") == NULL);

    size_t n;
    unsigned char* small = read_file(DIR "small.knx", &n);
    CHECK(rejected(small, n / 2));                        /* 잘린 파일 */
    CHECK(rejected(small, n + 4));                        /* 끝에 남는 바이트 (read_file이 4바이트 여유 할당) */
    small[3] = '2';
    CHECK(rejected(small, n));                            /* 매직 불일치 */
    free(small);

    uint32_t zero_ops[] = { MAGIC, 0 };
    uint32_t many_ops[] = { MAGIC, 1025 };
    uint32_t huge[] = { MAGIC, 1, 0, 0xFFFFFFFFu, 0xFFFFFFFFu, 1 };
    uint32_t big[] = { MAGIC, 1, 0, 30000, 30000, 0 };    /* 유효한 int지만 데이터 없음: 3.6GB 할당 전에 거부 */
    uint32_t zero_dim[] = { MAGIC, 1, 0, 0, 4, 0 };
    uint32_t chain[2 + 4 + 32 + 4 + 18] = { MAGIC, 2, 0, 4, 8, 0 }; /* Linear(4,8) 뒤 Linear(9,2) */
    chain[38] = 0; chain[39] = 9; chain[40] = 2; chain[41] = 0;
    uint32_t ln[2 + 4 + 32 + 4] = { MAGIC, 2, 0, 4, 8, 0 };         /* Linear(4,8) 뒤 LayerNorm(7) */
    ln[38] = 2; ln[39] = 7; ln[40] = 0; ln[41] = 0x3727C5ACu;       /* eps 1e-5 */
    uint32_t bad_kind[] = { MAGIC, 1, 9 };
    uint32_t bad_act[] = { MAGIC, 1, 1, 9 };
    uint32_t act_only[] = { MAGIC, 1, 1, 0 };             /* 차원을 가진 연산 없음 */
    CHECK(rejected(zero_ops, sizeof zero_ops));
    CHECK(rejected(many_ops, sizeof many_ops));
    CHECK(rejected(huge, sizeof huge));
    CHECK(rejected(big, sizeof big));
    CHECK(rejected(zero_dim, sizeof zero_dim));
    CHECK(rejected(chain, sizeof chain));
    CHECK(rejected(ln, sizeof ln));
    CHECK(rejected(bad_kind, sizeof bad_kind));
    CHECK(rejected(bad_act, sizeof bad_act));
    CHECK(rejected(act_only, sizeof act_only));
    chain[39] = 8;                                        /* 같은 파일을 Linear(8,2)로 고치면 통과해야 함 */
    write_file(DIR "bad.knx", chain, sizeof chain - 2 * 4);
    m = kernix_model_load(DIR "bad.knx");
    CHECK(m != NULL && m->in_dim == 4 && m->out_dim == 2 && m->n_steps == 2);
    kernix_model_free(m);
    printf("  로더 %s\n", fail ? "실패" : "통과 ✓");
}

/* mlp_bench.py의 save_tensor 형식: uint32 rows, cols, float32 데이터 */
static float_tensor* load_tensor(const char* path)
{
    size_t n;
    unsigned char* p = read_file(path, &n);
    uint32_t rc[2];
    memcpy(rc, p, 8);
    float_tensor* t = kernix_float_new((int)rc[0], (int)rc[1], 0);
    if (t == NULL || n != 8 + sizeof(float) * (size_t)rc[0] * rc[1]) { fprintf(stderr, "텐서 파일 오류: %s\n", path); exit(1); }
    memcpy(t->tensor, p + 8, n - 8);
    free(p);
    return t;
}

/* 모양이 다르거나 NaN이 있으면 INFINITY */
static float max_abs(const float_tensor* got, const float_tensor* want)
{
    if (got == NULL || got->row != want->row || got->col != want->col) return INFINITY;
    float worst = 0.0f;
    for (size_t i = 0; i < (size_t)want->row * want->col; i++) {
        float d = fabsf(got->tensor[i] - want->tensor[i]);
        if (!(d <= worst)) worst = d;
    }
    return worst == worst ? worst : INFINITY;
}

/* PyTorch 기대 출력과 비교(최대 절대 오차 1e-4), 입력이 바뀌지 않았는지도 확인 */
static void check_model(kernix_model* m, const char* name, int batch)
{
    char px[128], py[128];
    snprintf(px, sizeof px, DIR "%s_x_%d.bin", name, batch);
    snprintf(py, sizeof py, DIR "%s_y_%d.bin", name, batch);
    float_tensor* x = load_tensor(px);
    float_tensor* want = load_tensor(py);
    size_t bytes = sizeof(float) * (size_t)x->row * x->col;
    float* before = malloc(bytes);
    memcpy(before, x->tensor, bytes);
    float d = max_abs(kernix_model_forward(m, x), want);
    printf("  %-14s 배치 %-5d 최대 절대 오차 %.2e %s\n", name, batch, d, d <= 1e-4f ? "✓" : "✗");
    if (!(d <= 1e-4f)) fail = 1;
    CHECK(memcmp(before, x->tensor, bytes) == 0);
    free(before); float_free(x); float_free(want);
}

static void check_forward(void)
{
    kernix_model* m = kernix_model_load(DIR "small.knx");
    if (m == NULL) { fail = 1; return; }
    check_model(m, "small", 1);
    check_model(m, "small", 37);
    check_model(m, "small", 4096);
    check_model(m, "small", 37); /* 배치 크기를 바꿨다 돌아와도 같은 결과 (버퍼 재할당) */

    /* 같은 배치의 다음 호출은 같은 버퍼를 재사용 (할당 없음) */
    float_tensor* x37 = load_tensor(DIR "small_x_37.bin");
    const float_tensor* y1 = kernix_model_forward(m, x37);
    const float* p1 = y1 ? y1->tensor : NULL;
    const float_tensor* y2 = kernix_model_forward(m, x37);
    CHECK(y1 != NULL && y1 == y2 && y2->tensor == p1);
    float_free(x37);

    float_tensor* x0 = float_zeros(0, 64);
    const float_tensor* y0 = kernix_model_forward(m, x0);
    CHECK(y0 != NULL && y0->row == 0 && y0->col == 10);
    float_free(x0);

    float_tensor* xbad = float_zeros(2, 63);
    CHECK(kernix_model_forward(m, xbad) == NULL); /* 열 수 불일치 */
    CHECK(kernix_model_forward(m, NULL) == NULL);
    CHECK(kernix_model_forward(NULL, xbad) == NULL);
    float_free(xbad);
    kernix_model_free(m);

    m = kernix_model_load(DIR "wide.knx"); /* 열 9000 > KERNIX_CHUNK: SiLU/GELU 임시 버퍼 경계 */
    if (m != NULL) check_model(m, "wide", 5); else fail = 1;
    kernix_model_free(m);

    m = kernix_model_load(DIR "inplace_first.knx"); /* 첫 단계가 제자리 연산: 입력 보존 */
    if (m != NULL) check_model(m, "inplace_first", 3); else fail = 1;
    kernix_model_free(m);

    /* 자기 출력을 입력으로 다시 넣으면 버퍼가 겹치므로 거부. Linear(4,4) 단위행렬 모델 */
    uint32_t sq[6 + 16] = { MAGIC, 1, 0, 4, 4, 0 };
    for (int i = 0; i < 4; i++) sq[6 + i * 5] = ONE;
    write_file(DIR "bad.knx", sq, sizeof sq);
    m = kernix_model_load(DIR "bad.knx");
    float_tensor* x4 = float_zeros(2, 4);
    for (int i = 0; i < 8; i++) x4->tensor[i] = (float)i;
    const float_tensor* y = m ? kernix_model_forward(m, x4) : NULL;
    CHECK(y != NULL && max_abs(y, x4) == 0.0f);
    CHECK(y != NULL && kernix_model_forward(m, y) == NULL);
    float_free(x4);
    kernix_model_free(m);
    printf("  forward %s\n", fail ? "실패" : "통과 ✓");
}

/* ── LSTM: 직접 만든 파일을 로드해 스칼라 참조 구현과 비교 ── */
static uint32_t fw[4096];
static size_t fwn;
static void put(uint32_t v) { fw[fwn++] = v; }
static void put_floats(size_t n) /* [-0.5, 0.5) 의사 난수 */
{
    static uint32_t seed = 12345;
    for (size_t i = 0; i < n; i++) {
        seed = seed * 1664525u + 1013904223u;
        float v = (float)(seed >> 8) / 16777216.0f - 0.5f;
        memcpy(&fw[fwn++], &v, 4);
    }
}
static void put_lstm(uint32_t in, uint32_t units, uint32_t act, uint32_t ret_seq, uint32_t seq)
{
    put(KERNIX_OP_LSTM); put(in); put(units); put(act); put(ret_seq); put(seq);
    put_floats((size_t)in * 4 * units + (size_t)units * 4 * units + 4 * units);
}

static float r_sig(float v) { return 1.0f / (1.0f + expf(-v)); }
static float r_act(int act, float v) { return act == KERNIX_ACT_TANH ? tanhf(v) : v > 0.0f ? v : 0.0f; }

/* Keras LSTM과 같은 정의(게이트 순서 i, f, g, o). x: B*T×in. out: ret_seq면 B*T×u, 아니면 B×u */
static void ref_lstm(const kernix_step* s, const float* x, int B, float* out)
{
    const int T = s->seq, u = s->out, in = s->in;
    for (int b = 0; b < B; b++) {
        float h[64] = { 0 }, c[64] = { 0 }, z[256];
        for (int t = 0; t < T; t++) {
            for (int j = 0; j < 4 * u; j++) {
                double acc = s->bias[j];
                for (int k = 0; k < in; k++) acc += (double)x[(b * T + t) * in + k] * s->w->tensor[k * 4 * u + j];
                for (int k = 0; k < u; k++) acc += (double)h[k] * s->rec[k * 4 * u + j];
                z[j] = (float)acc;
            }
            for (int k = 0; k < u; k++) {
                c[k] = r_sig(z[u + k]) * c[k] + r_sig(z[k]) * r_act(s->act, z[2 * u + k]);
                h[k] = r_sig(z[3 * u + k]) * r_act(s->act, c[k]);
                if (s->ret_seq) out[(b * T + t) * u + k] = h[k];
            }
        }
        if (!s->ret_seq) memcpy(out + b * u, h, sizeof(float) * (size_t)u);
    }
}

/* LSTM(7→5, tanh, 시퀀스 반환, T=4) → LSTM(5→6, relu, 마지막만) → Linear(6→3) 모델 파일을 path에 씀 */
static void write_lstm_model(const char* path)
{
    fwn = 0;
    put(MAGIC); put(3);
    put_lstm(7, 5, KERNIX_ACT_TANH, 1, 4);
    put_lstm(5, 6, KERNIX_ACT_RELU, 0, 4);
    put(KERNIX_OP_LINEAR); put(6); put(3); put(0); put_floats(18);
    write_file(path, fw, fwn * 4);
}

static void check_lstm(void)
{
    write_lstm_model(DIR "bad.knx");
    kernix_model* m = kernix_model_load(DIR "bad.knx");
    CHECK(m != NULL && m->n_steps == 3 && m->in_dim == 7 && m->out_dim == 3 && m->seq_len == 4);
    if (m != NULL) {
        const int B = 3;
        float_tensor* x = float_zeros(B * 4, 7);
        for (int i = 0; i < B * 4 * 7; i++) x->tensor[i] = (float)((i * 37) % 23 - 11) * 0.1f;
        float h1[3 * 4 * 5], h2[3 * 6];
        float_tensor want = { B, 3, (float[9]){ 0 } };
        ref_lstm(&m->steps[0], x->tensor, B, h1);
        ref_lstm(&m->steps[1], h1, B, h2);
        for (int b = 0; b < B; b++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 6; k++) want.tensor[b * 3 + j] += h2[b * 6 + k] * m->steps[2].w->tensor[k * 3 + j];
        for (int rep = 0; rep < 2; rep++) { /* 두 번째 호출: 재사용 버퍼에 남은 상태(h, c)가 새지 않아야 함 */
            float d = max_abs(kernix_model_forward(m, x), &want);
            printf("  LSTM 2층 + Linear   최대 절대 오차 %.2e %s\n", d, d <= 1e-5f ? "✓" : "✗");
            if (!(d <= 1e-5f)) fail = 1;
        }
        float_tensor* x7 = float_zeros(7, 7); /* 행 수가 T의 배수가 아님 */
        CHECK(kernix_model_forward(m, x7) == NULL);
        float_free(x7); float_free(x);
    }
    kernix_model_free(m);

    /* 불량 LSTM 파일 */
    fwn = 0; put(MAGIC); put(1); put_lstm(7, 5, KERNIX_ACT_TANH, 0, 0);       /* seq_len 0 */
    CHECK(rejected(fw, fwn * 4));
    fwn = 0; put(MAGIC); put(1); put_lstm(7, 5, KERNIX_ACT_GELU, 0, 4);       /* relu/tanh만 */
    CHECK(rejected(fw, fwn * 4));
    fwn = 0; put(MAGIC); put(1); put_lstm(7, 5, KERNIX_ACT_TANH, 2, 4);       /* ret_seq는 0/1 */
    CHECK(rejected(fw, fwn * 4));
    fwn = 0; put(MAGIC); put(2); put_lstm(7, 5, KERNIX_ACT_TANH, 0, 4); put_lstm(5, 6, KERNIX_ACT_RELU, 0, 4);
    CHECK(rejected(fw, fwn * 4));                                             /* 시퀀스가 끝난 뒤 또 LSTM */
    fwn = 0; put(MAGIC); put(1); put(KERNIX_OP_LSTM); put(7); put(100000); put(KERNIX_ACT_RELU); put(0); put(4);
    CHECK(rejected(fw, fwn * 4));                                             /* units 과대, 데이터 없음 */
    fwn = 0; put(MAGIC); put(1); put_lstm(7, 5, KERNIX_ACT_TANH, 0, 4);
    CHECK(rejected(fw, fwn * 4 - 4));                                         /* 잘림 */
    printf("  LSTM %s\n", fail ? "실패" : "통과 ✓");
}

/* 표본을 나눠 실행해도 한 번에 실행한 결과와 같은지, 마지막 조각 뒤 버퍼 재사용, trim 후 재할당 */
static void check_chunk_model(kernix_model* m, const float_tensor* x, const char* name, float tol)
{
    const int samples = x->row / m->seq_len, T = m->seq_len;
    const size_t per = m->ws_per_sample * sizeof(float), fixed = m->ws_fixed * sizeof(float);
    kernix_model_set_workspace_limit(m, 0);
    const float_tensor* y = kernix_model_forward(m, x);
    CHECK(y != NULL && m->batch == x->row);
    if (y == NULL) return;
    float_tensor* want = float_zeros(y->row, y->col);
    memcpy(want->tensor, y->tensor, sizeof(float) * (size_t)y->row * (size_t)y->col);

    const size_t limits[] = { fixed + 1, fixed + per * 3, fixed + per * (size_t)(samples - 1) }; /* 1개씩, 3개씩 + 나머지, 마지막 조각 1개 */
    const int chunks[] = { 1, 3, samples - 1 };
    float worst = 0.0f;
    for (int i = 0; i < 3; i++) {
        kernix_model_set_workspace_limit(m, limits[i]);
        y = kernix_model_forward(m, x);
        CHECK(y != NULL && y->row == samples && y->col == m->out_dim && m->batch == chunks[i] * T);
        float d = y != NULL ? max_abs(y, want) : 1.0f;
        worst = d > worst ? d : worst;
    }
    /* 3개씩 나눠 10개(3,3,3,1)를 돌린 뒤 3개만: 할당 없이 같은 버퍼를 원래 행 수로 되돌려 써야 함 */
    kernix_model_set_workspace_limit(m, fixed + per * 3);
    kernix_model_forward(m, x);
    const float_tensor x3 = { 3 * T, x->col, x->tensor }, want3 = { 3, want->col, want->tensor };
    y = kernix_model_forward(m, &x3);
    CHECK(y != NULL && m->batch == 3 * T);
    float d3 = y != NULL ? max_abs(y, &want3) : 1.0f;

    kernix_model_trim(m);
    int empty = m->batch == -1 && m->in_copy == NULL && m->out_all == NULL;
    for (int i = 0; i < m->n_steps; i++) empty &= m->steps[i].buf == NULL && m->steps[i].tmp[0] == NULL && m->steps[i].deq == NULL;
    CHECK(empty);
    y = kernix_model_forward(m, x);
    float dt = y != NULL ? max_abs(y, want) : 1.0f;
    const int ok = worst <= tol && d3 <= tol && dt <= tol;
    printf("  %-18s 나눠 실행 최대 오차 %.1e, 조각 뒤 재사용 %.1e, trim 후 %.1e %s\n", name, worst, d3, dt, ok ? "✓" : "✗");
    if (!ok) fail = 1;
    float_free(want);
}

static void check_chunk(void)
{
    float_tensor* x = float_zeros(10, 8); /* 첫 단계가 제자리(Tanh)라 입력 복사본을 씀 */
    for (int i = 0; i < 80; i++) x->tensor[i] = (float)((i * 29) % 17 - 8) * 0.2f;
    kernix_model* m = kernix_model_load(DIR "inplace_first.knx");
    if (m != NULL) check_chunk_model(m, x, "inplace_first", 1e-5f);
    kernix_model_free(m);
    float_free(x);

    float_tensor* xs = load_tensor(DIR "small_x_37.bin");
    m = kernix_model_load(DIR "small.knx");
    if (m != NULL && xs != NULL) check_chunk_model(m, xs, "small (MLP)", 1e-5f);
    kernix_model_free(m);
    float_free(xs);

    write_lstm_model(DIR "chunk_lstm.knx"); /* bad.knx가 아닌 파일: 다른 테스트가 bad.knx를 불량 파일로 기대함 */
    x = float_zeros(10 * 4, 7); /* 시퀀스 10개 */
    for (int i = 0; i < 10 * 4 * 7; i++) x->tensor[i] = (float)((i * 37) % 23 - 11) * 0.1f;
    m = kernix_model_load(DIR "chunk_lstm.knx");
    if (m != NULL) check_chunk_model(m, x, "LSTM 2층", 1e-5f);
    kernix_model_free(m);
    float_free(x);
    printf("  나눠 실행 / trim %s\n", fail ? "실패" : "통과 ✓");
}

/* int8 커널 참조: 같은 int8 값으로 스칼라 계산 (int64 누적). 입력 양자화는 커널과 같은 x·(1/sx) */
static void ref_qmatmul(const float* x, int rows, int k, int n, const int8_t* q, const float* scale,
                        const float* bias, int accumulate, float* y)
{
    for (int i = 0; i < rows; i++) {
        const float* xr = x + (size_t)i * k;
        float m = 0.0f;
        for (int j = 0; j < k; j++) m = fmaxf(m, fabsf(xr[j]));
        const float sx = m > 0.0f ? m / 127.0f : 1.0f, inv = 1.0f / sx;
        for (int c = 0; c < n; c++) {
            int64_t s = 0;
            for (int j = 0; j < k; j++) {
                float v = rintf(xr[j] * inv);
                v = v > 127.0f ? 127.0f : v < -127.0f ? -127.0f : v;
                s += (int64_t)v * q[(size_t)j * n + c];
            }
            float r = (float)s * sx * scale[c] + (bias ? bias[c] : 0.0f);
            y[(size_t)i * n + c] = accumulate ? y[(size_t)i * n + c] + r : r;
        }
    }
}

static uint32_t qrng = 12345;
static uint32_t qrand(void) { qrng = qrng * 1664525u + 1013904223u; return qrng >> 8; }

static void check_qmatmul(void)
{
    const int cases[][3] = { { 55, 31, 1 }, { 64, 256, 1 }, { 128, 512, 3 }, { 114, 512, 32 }, { 16, 16, 1 }, { 1, 1, 1 } };
    float worst = 0.0f;
    int unpack_ok = 1;
    for (int c = 0; c < 6; c++) {
        const int k = cases[c][0], n = cases[c][1], rows = cases[c][2];
        int8_t* q = malloc((size_t)k * n);
        float *scale = malloc(sizeof(float) * n), *bias = malloc(sizeof(float) * n), *x = malloc(sizeof(float) * rows * k);
        float *y = malloc(sizeof(float) * rows * n), *want = malloc(sizeof(float) * rows * n), *deq = malloc(sizeof(float) * k * n);
        for (size_t i = 0; i < (size_t)k * n; i++) q[i] = (int8_t)((int)(qrand() % 255) - 127);
        for (int i = 0; i < n; i++) { scale[i] = 0.001f + (float)(qrand() % 1000) * 1e-5f; bias[i] = (float)(qrand() % 200) * 0.01f - 1.0f; }
        for (int i = 0; i < rows * k; i++) x[i] = (float)(qrand() % 2001) * 0.003f - 3.0f;
        kernix_qweight* w = kernix_qweight_pack(q, scale, k, n);
        CHECK(w != NULL && w->k == k && w->n == n && w->kp % 16 == 0 && w->np % 16 == 0 && w->kp >= k && w->np >= n);
        if (w == NULL) continue;
        kernix_qweight_unpack(w, deq);
        for (size_t i = 0; i < (size_t)k * n; i++) unpack_ok &= deq[i] == (float)q[i] * scale[i % n];
        for (int acc = 0; acc < 2; acc++) {
            for (int i = 0; i < rows * n; i++) y[i] = want[i] = (float)(i % 7) * 0.5f; /* accumulate=1이면 이 값에 더해짐 */
            CHECK(kernix_qmatmul(x, rows, w, bias, acc, y) == 0);
            ref_qmatmul(x, rows, k, n, q, scale, bias, acc, want);
            float mx = 1e-30f, d = 0.0f;
            for (int i = 0; i < rows * n; i++) { mx = fmaxf(mx, fabsf(want[i])); d = fmaxf(d, fabsf(y[i] - want[i])); }
            worst = fmaxf(worst, d / mx);
        }
        if (rows >= 3) { /* 1행에 NaN 하나: 그 행만 전부 NaN. 2행은 0: 정확히 bias */
            x[k] = NAN;
            for (int j = 0; j < k; j++) x[2 * k + j] = 0.0f;
            CHECK(kernix_qmatmul(x, rows, w, bias, 0, y) == 0);
            int nan_ok = 1, other_ok = 1, zero_ok = 1;
            ref_qmatmul(x, 1, k, n, q, scale, bias, 0, want);
            for (int j = 0; j < n; j++) {
                nan_ok &= isnan(y[n + j]);
                other_ok &= fabsf(y[j] - want[j]) <= 1e-5f * fmaxf(1.0f, fabsf(want[j]));
                zero_ok &= y[2 * n + j] == bias[j];
            }
            CHECK(nan_ok && other_ok && zero_ok);
        }
        kernix_qweight_free(w);
        free(q); free(scale); free(bias); free(x); free(y); free(want); free(deq);
    }
    CHECK(unpack_ok);
    /* −128 경계: K = 131072, q = −128, x = 1 → 128·127·K < 2³¹ 이라 int32에서 넘치지 않음 */
    const int bk = KERNIX_INT8_MAX_K;
    int8_t* q = malloc(bk);
    float* x = malloc(sizeof(float) * bk);
    for (int i = 0; i < bk; i++) { q[i] = -128; x[i] = 1.0f; }
    const float one = 1.0f;
    kernix_qweight* w = kernix_qweight_pack(q, &one, bk, 1);
    float y = 0.0f;
    CHECK(w != NULL && kernix_qmatmul(x, 1, w, NULL, 0, &y) == 0);
    CHECK(y == (float)(-128.0 * 127.0 * bk) * (1.0f / 127.0f));
    kernix_qweight_free(w);
    free(q); free(x);
    const int ok = worst <= 1e-5f && !fail;
    printf("  int8 커널: 참조 대비 최대 상대 오차 %.1e %s\n", worst, worst <= 1e-5f ? "✓" : "✗");
    if (!(worst <= 1e-5f)) fail = 1;
    printf("  int8 커널 %s\n", ok ? "통과 ✓" : "실패");
}

/* ---- .knx v2 (int8 가중치) ---- 바이트 단위 기록: int8 값 때문에 4바이트 정렬이 아님 */
#define MAGIC2 0x32584E4Bu /* "KNX2" */
enum { DT_F32, DT_I8COL, DT_I8ROW };
static unsigned char vb[1 << 21];
static size_t vbn;
static void vput(uint32_t v) { memcpy(vb + vbn, &v, 4); vbn += 4; }
static void vputf(float v) { memcpy(vb + vbn, &v, 4); vbn += 4; }
static void vput8(int8_t v) { vb[vbn++] = (unsigned char)v; }
static float vrandf(void) { return (float)(qrand() % 2001) * 0.0005f - 0.5f; }
static void vput_f32(int k, int n, float* keep) /* F32 블록, 값은 keep에도 (NULL 가능) */
{
    vput(DT_F32);
    for (int i = 0; i < k * n; i++) { float v = vrandf(); if (keep) keep[i] = v; vputf(v); }
}
static void vput_i8(int dt, int k, int n, int8_t* q, float* scale) /* I8 블록, q(k×n)와 scale을 기록 */
{
    const int ns = dt == DT_I8COL ? n : k;
    vput((uint32_t)dt);
    for (int i = 0; i < ns; i++) { scale[i] = 0.01f + (float)(qrand() % 100) * 1e-4f; vputf(scale[i]); }
    for (int i = 0; i < k * n; i++) { q[i] = (int8_t)((int)(qrand() % 255) - 127); vput8(q[i]); }
}
static void vlinear_head(uint32_t in, uint32_t out, uint32_t bias) { vput(KERNIX_OP_LINEAR); vput(in); vput(out); vput(bias); }

static void check_v2_load(void)
{
    int8_t q[4096];
    float sc[4096];
    /* 거부: 머리 KNX3, dtype 3, 잘린 scale, 잘린 int8, scale 0 / -1 / NaN / inf, K = 131073 */
    vbn = 0; vput(0x33584E4Bu); vput(1); vlinear_head(4, 2, 0); vput_f32(4, 2, NULL);
    CHECK(rejected(vb, vbn));
    vbn = 0; vput(MAGIC2); vput(1); vlinear_head(4, 2, 0); vput(3); for (int i = 0; i < 8; i++) vputf(0.1f);
    CHECK(rejected(vb, vbn));
    vbn = 0; vput(MAGIC2); vput(1); vlinear_head(4, 2, 0); vput(DT_I8COL); vputf(0.1f);
    CHECK(rejected(vb, vbn));
    vbn = 0; vput(MAGIC2); vput(1); vlinear_head(4, 2, 0); vput_i8(DT_I8COL, 4, 2, q, sc);
    CHECK(rejected(vb, vbn - 1));
    const float bad[4] = { 0.0f, -1.0f, NAN, INFINITY };
    for (int b = 0; b < 4; b++) {
        vbn = 0; vput(MAGIC2); vput(1); vlinear_head(4, 2, 0); vput_i8(DT_I8COL, 4, 2, q, sc);
        memcpy(vb + 4 * 2 + 4 * 4 + 4 + 4, &bad[b], 4); /* 머리 8 + Linear 머리 16 + dtype 4 + 두 번째 scale */
        CHECK(rejected(vb, vbn));
    }
    vbn = 0; vput(MAGIC2); vput(1); vlinear_head(KERNIX_INT8_MAX_K + 1, 1, 0); vput(DT_I8COL); vputf(0.01f);
    for (int i = 0; i < KERNIX_INT8_MAX_K + 1; i++) vput8(1);
    CHECK(rejected(vb, vbn));
    vbn = 0; vput(MAGIC2); vput(1); vlinear_head(KERNIX_INT8_MAX_K, 1, 0); vput(DT_I8COL); vputf(0.01f);
    for (int i = 0; i < KERNIX_INT8_MAX_K; i++) vput8(1);
    write_file(DIR "v2.knx", vb, vbn);
    kernix_model* m = kernix_model_load(DIR "v2.knx"); /* 상한 정확히: 받아들임 */
    CHECK(m != NULL && m->steps[0].q != NULL);
    kernix_model_free(m);

    /* 받아들임: 모두 0인 열(scale 1)은 bias만 남음 */
    const int8_t zq[8] = { 10, 0, 20, 0, 30, 0, 40, 0 };
    const float zs[2] = { 0.02f, 1.0f }, zb[2] = { 0.25f, -0.75f };
    vbn = 0; vput(MAGIC2); vput(1); vlinear_head(4, 2, 1); vput(DT_I8COL); vputf(zs[0]); vputf(zs[1]);
    for (int i = 0; i < 8; i++) vput8(zq[i]);
    vputf(zb[0]); vputf(zb[1]);
    write_file(DIR "v2.knx", vb, vbn);
    m = kernix_model_load(DIR "v2.knx");
    CHECK(m != NULL && m->steps[0].q != NULL && m->steps[0].w == NULL);
    if (m != NULL) {
        float xv[4] = { 1.0f, -2.0f, 0.5f, 3.0f }, want[2];
        float_tensor x = { 1, 4, xv };
        const float_tensor* y = kernix_model_forward(m, &x);
        ref_qmatmul(xv, 1, 4, 2, zq, zs, zb, 0, want);
        CHECK(y != NULL && y->tensor[1] == -0.75f && fabsf(y->tensor[0] - want[0]) <= 1e-6f);
    }
    kernix_model_free(m);

    /* I8_ROW Linear는 float로 풀림: w[k][n] = q·scale[k] */
    vbn = 0; vput(MAGIC2); vput(1); vlinear_head(3, 5, 0); vput_i8(DT_I8ROW, 3, 5, q, sc);
    write_file(DIR "v2.knx", vb, vbn);
    m = kernix_model_load(DIR "v2.knx");
    int row_ok = m != NULL && m->steps[0].q == NULL && m->steps[0].w != NULL;
    for (int i = 0; row_ok && i < 15; i++) row_ok = m->steps[0].w->tensor[i] == (float)q[i] * sc[i / 5];
    CHECK(row_ok);
    kernix_model_free(m);

    /* LSTM 입력 가중치 I8_COL은 float로 풀리고, 순환 F32는 float 그대로 */
    vbn = 0; vput(MAGIC2); vput(1); vput(KERNIX_OP_LSTM); vput(3); vput(2); vput(KERNIX_ACT_TANH); vput(0); vput(2);
    vput_i8(DT_I8COL, 3, 8, q, sc); vput_f32(2, 8, NULL); for (int i = 0; i < 8; i++) vputf(0.0f);
    write_file(DIR "v2.knx", vb, vbn);
    m = kernix_model_load(DIR "v2.knx");
    int lstm_ok = m != NULL && m->steps[0].q == NULL && m->steps[0].rec != NULL && m->steps[0].w != NULL;
    for (int i = 0; lstm_ok && i < 24; i++) lstm_ok = m->steps[0].w->tensor[i] == (float)q[i] * sc[i % 8];
    CHECK(lstm_ok);
    kernix_model_free(m);
    printf("  v2 로더 %s\n", fail ? "실패" : "통과 ✓");
}

/* LSTM(7→5, tanh, 시퀀스, T=4) → LSTM(5→6, relu) → Linear(6→3), 순환과 Linear는 I8_COL. 값은 참조용으로 보관 */
typedef struct { float w1[7 * 20], b1[20], w2[5 * 24], b2[24], bd[3]; int8_t u1[5 * 20], u2[6 * 24], d[6 * 3]; float s1[20], s2[24], sd[3]; } qlstm_ref;
static void write_qlstm_model(const char* path, qlstm_ref* r)
{
    vbn = 0; vput(MAGIC2); vput(3);
    vput(KERNIX_OP_LSTM); vput(7); vput(5); vput(KERNIX_ACT_TANH); vput(1); vput(4);
    vput_f32(7, 20, r->w1); vput_i8(DT_I8COL, 5, 20, r->u1, r->s1); for (int i = 0; i < 20; i++) vputf(r->b1[i] = vrandf());
    vput(KERNIX_OP_LSTM); vput(5); vput(6); vput(KERNIX_ACT_RELU); vput(0); vput(4);
    vput_f32(5, 24, r->w2); vput_i8(DT_I8COL, 6, 24, r->u2, r->s2); for (int i = 0; i < 24; i++) vputf(r->b2[i] = vrandf());
    vlinear_head(6, 3, 1); vput_i8(DT_I8COL, 6, 3, r->d, r->sd); for (int i = 0; i < 3; i++) vputf(r->bd[i] = vrandf());
    write_file(path, vb, vbn);
}

/* 참조: deq = 0이면 순환·Linear를 ref_qmatmul(int8), 1이면 q·scale로 푼 float로 (double 누적) */
static void ref_qlstm_layer(const float* x, int B, int in, int u, int ret, const float* w, const float* b,
                            const int8_t* q, const float* sc, int deq, float* out)
{
    const int T = 4, g = 4 * u;
    for (int s = 0; s < B; s++) {
        float h[8] = { 0 }, c[8] = { 0 }, z[32], hu[32];
        for (int t = 0; t < T; t++) {
            for (int j = 0; j < g; j++) {
                double a = b[j];
                for (int k = 0; k < in; k++) a += (double)x[(s * T + t) * in + k] * w[k * g + j];
                if (deq) for (int k = 0; k < u; k++) a += (double)h[k] * ((float)q[k * g + j] * sc[j]);
                z[j] = (float)a;
            }
            if (!deq && t > 0) { ref_qmatmul(h, 1, u, g, q, sc, NULL, 0, hu); for (int j = 0; j < g; j++) z[j] += hu[j]; }
            for (int k = 0; k < u; k++) {
                const int act = u == 5 ? KERNIX_ACT_TANH : KERNIX_ACT_RELU;
                c[k] = r_sig(z[u + k]) * c[k] + r_sig(z[k]) * r_act(act, z[2 * u + k]);
                h[k] = r_sig(z[3 * u + k]) * r_act(act, c[k]);
                if (ret) out[(s * T + t) * u + k] = h[k];
            }
        }
        if (!ret) memcpy(out + s * u, h, sizeof(float) * (size_t)u);
    }
}
static void ref_qlstm(const qlstm_ref* r, const float* x, int B, int deq, float* y)
{
    float h1[40 * 4 * 5], h2[40 * 6];
    ref_qlstm_layer(x, B, 7, 5, 1, r->w1, r->b1, r->u1, r->s1, deq, h1);
    ref_qlstm_layer(h1, B, 5, 6, 0, r->w2, r->b2, r->u2, r->s2, deq, h2);
    for (int s = 0; s < B; s++) {
        if (deq) {
            for (int j = 0; j < 3; j++) {
                double a = r->bd[j];
                for (int k = 0; k < 6; k++) a += (double)h2[s * 6 + k] * ((float)r->d[k * 3 + j] * r->sd[j]);
                y[s * 3 + j] = (float)a;
            }
        } else {
            ref_qmatmul(h2 + s * 6, 1, 6, 3, r->d, r->sd, r->bd, 0, y + s * 3);
        }
    }
}

static void check_v2_forward(void)
{
    /* v1 = v2 F32: 같은 바이트의 가중치에 dtype 0만 끼운 파일은 비트 단위로 같은 출력 */
    write_lstm_model(DIR "chunk_lstm.knx");
    vbn = 0; vput(MAGIC2); vput(3);
    size_t p = 2;
    const size_t parts[3][4] = { { 6, 7 * 20, 5 * 20, 20 }, { 6, 5 * 24, 6 * 24, 24 }, { 4, 6 * 3, 0, 0 } };
    for (int op = 0; op < 3; op++) {
        for (size_t i = 0; i < parts[op][0]; i++) vput(fw[p++]);
        for (int t = 1; t < 4; t++) {
            if (parts[op][t] == 0) continue;
            if (t < 3) vput(DT_F32); /* bias는 블록이 아님 */
            for (size_t i = 0; i < parts[op][t]; i++) vput(fw[p++]);
        }
    }
    write_file(DIR "v2.knx", vb, vbn); /* bad.knx는 불량 파일 테스트(Python 포함)가 거부되길 기대하므로 다른 이름 */
    kernix_model *m1 = kernix_model_load(DIR "chunk_lstm.knx"), *m2 = kernix_model_load(DIR "v2.knx");
    float_tensor* x = float_zeros(40 * 4, 7);
    for (int i = 0; i < 40 * 4 * 7; i++) x->tensor[i] = (float)((i * 37) % 23 - 11) * 0.1f;
    CHECK(m1 != NULL && m2 != NULL);
    if (m1 != NULL && m2 != NULL) {
        const float_tensor *y1 = kernix_model_forward(m1, x), *y2 = kernix_model_forward(m2, x);
        CHECK(y1 != NULL && y2 != NULL && memcmp(y1->tensor, y2->tensor, sizeof(float) * 40 * 3) == 0);
    }
    kernix_model_free(m1);
    kernix_model_free(m2);

    /* int8 실행: B = 1, MAX_ROWS (int8 경로), MAX_ROWS+1, 40 (float 사본 경로) */
    static qlstm_ref r;
    write_qlstm_model(DIR "qlstm.knx", &r);
    kernix_model* m = kernix_model_load(DIR "qlstm.knx");
    CHECK(m != NULL && m->steps[0].q != NULL && m->steps[1].q != NULL && m->steps[2].q != NULL && m->ws_fixed == 5 * 20 + 6 * 24 + 6 * 3);
    if (m != NULL) {
        float want[40 * 3], y40[40 * 3];
        const int Bs[4] = { 1, KERNIX_INT8_MAX_ROWS, KERNIX_INT8_MAX_ROWS + 1, 40 };
        float err_i8 = 0.0f, err_deq = 0.0f;
        for (int bi = 0; bi < 4; bi++) {
            const int B = Bs[bi], deq = B > KERNIX_INT8_MAX_ROWS;
            const float_tensor xb = { B * 4, 7, x->tensor };
            const float_tensor* y = kernix_model_forward(m, &xb);
            CHECK(y != NULL && y->row == B);
            if (y == NULL) continue;
            ref_qlstm(&r, x->tensor, B, deq, want);
            float d = 0.0f;
            for (int i = 0; i < B * 3; i++) d = fmaxf(d, fabsf(y->tensor[i] - want[i]));
            if (deq) err_deq = fmaxf(err_deq, d); else err_i8 = fmaxf(err_i8, d);
            if (B == 40) memcpy(y40, y->tensor, sizeof y40);
            for (int i = 0; i < m->n_steps; i++) /* 사본은 할당 행 수가 임계값을 넘을 때만 */
                if (m->steps[i].q != NULL) CHECK((m->steps[i].deq != NULL) == deq);
        }
        float path = 0.0f; /* 같은 표본을 1개씩(int8) vs 40개 한 번에(사본) */
        for (int s = 0; s < 40; s++) {
            const float_tensor xs = { 4, 7, x->tensor + s * 4 * 7 };
            const float_tensor* y = kernix_model_forward(m, &xs);
            for (int j = 0; y != NULL && j < 3; j++) path = fmaxf(path, fabsf(y->tensor[j] - y40[s * 3 + j]));
        }
        const int ok = err_i8 <= 1e-5f && err_deq <= 1e-5f && path <= 1e-2f;
        printf("  int8 LSTM 2층 + Linear: int8 경로 %.1e, 사본 경로 %.1e, 두 경로 차이 %.1e %s\n", err_i8, err_deq, path, ok ? "✓" : "✗");
        if (!ok) fail = 1;

        check_chunk_model(m, x, "int8 LSTM", 1e-2f);
        kernix_model_set_workspace_limit(m, m->ws_fixed * sizeof(float) + m->ws_per_sample * sizeof(float) * 3);
        const float_tensor x10 = { 10 * 4, 7, x->tensor };
        CHECK(kernix_model_forward(m, &x10) != NULL && m->batch == 3 * 4);
        for (int i = 0; i < m->n_steps; i++)
            if (m->steps[i].q != NULL) CHECK(m->steps[i].deq != NULL); /* 조각 3개 > 임계값 2: 사본 필요 */
    }
    kernix_model_free(m);
    float_free(x);
    printf("  v2 실행 %s\n", fail ? "실패" : "통과 ✓");
}

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* 프로세스의 모든 스레드(GCD 작업자 포함)가 쓴 CPU 시간(초) */
static double cpu_sec(void)
{
    struct rusage u;
    getrusage(RUSAGE_SELF, &u);
    return (double)(u.ru_utime.tv_sec + u.ru_stime.tv_sec) + (double)(u.ru_utime.tv_usec + u.ru_stime.tv_usec) * 1e-6;
}

static int cmp_double(const void* x, const void* y)
{
    double a = *(const double*)x, b = *(const double*)y;
    return (a > b) - (a < b);
}

/* 런타임 도입 전 방식: 기존 연산을 이어 붙임. 레이어마다 텐서 3개 할당, 메모리 3번 왕복 (Linear+ReLU 모델 전용) */
static float_tensor* naive_forward(const kernix_model* m, const float_tensor* x)
{
    const float_tensor* cur = x;
    for (int i = 0; i < m->n_steps; i++) {
        const kernix_step* s = &m->steps[i];
        float_tensor bias = { 1, s->out, s->bias };
#if defined(KERNIX_USE_METAL)
        float_tensor* y = float_matmul_metal(cur, s->w);
#else
        float_tensor* y = float_matmul_parallel(cur, s->w);
#endif
        float_tensor* t = floats_add(y, &bias);
        float_free(y);
        if (s->act == KERNIX_ACT_RELU) { y = floats_relu(t); float_free(t); t = y; }
        if (cur != x) float_free((float_tensor*)cur);
        cur = t;
    }
    return (float_tensor*)cur;
}

#define RUNS 20

/* 출력: "KERNIX <variant> <배치> <forward 중앙값 ms> <forward당 CPU ms>", 끝에 프로세스 최대 메모리 */
static int bench(const char* variant)
{
    const int naive = strcmp(variant, "naive") == 0;
    kernix_model* m = kernix_model_load(DIR "bench.knx");
    if (m == NULL) return 1;
    const int batches[] = { 1, 64, 4096 };
    for (int b = 0; b < 3; b++) {
        float_tensor* x = float_zeros(batches[b], m->in_dim);
        if (x == NULL) return 1;
        for (size_t i = 0; i < (size_t)x->row * x->col; i++) x->tensor[i] = (float)((int)(i % 200) - 100) * 0.01f;
        double t[RUNS];
        if (naive) float_free(naive_forward(m, x)); else kernix_model_forward(m, x); /* 워밍업 */
        double c0 = cpu_sec();
        for (int i = 0; i < RUNS; i++) {
            double st = now();
            if (naive) float_free(naive_forward(m, x));
            else if (kernix_model_forward(m, x) == NULL) return 1;
            t[i] = (now() - st) * 1e3;
        }
        double cpu = (cpu_sec() - c0) * 1e3 / RUNS;
        qsort(t, RUNS, sizeof(double), cmp_double);
        printf("KERNIX %s %d %.4f %.4f\n", variant, batches[b], (t[RUNS / 2 - 1] + t[RUNS / 2]) / 2.0, cpu);
        float_free(x);
    }
    struct rusage u;
    getrusage(RUSAGE_SELF, &u);
    printf("KERNIX %s maxrss_mb %.1f\n", variant, (double)u.ru_maxrss / (1 << 20)); /* macOS는 바이트 단위 */
    kernix_model_free(m);
    return 0;
}

/* 외부 모델 평가 (test/sign_bench.py가 호출). <name>.knx, <name>_x.bin(입력), <name>_y.bin(원본 프레임워크 출력),
 * <name>_labels.bin(정답)을 읽어 "키 값..." 줄을 출력: 정확성, 표본 1개 지연, 전체 배치 시간, 최대 메모리 */
/* 모델 이름에서 데이터 이름: "<이름>_int8"은 float 모델과 같은 검증 데이터(<이름>_x.bin 등)를 씀 */
static const char* data_name(const char* model, char* buf, size_t n)
{
    const size_t len = strlen(model);
    snprintf(buf, n, "%.*s", (int)(len > 5 && strcmp(model + len - 5, "_int8") == 0 ? len - 5 : len), model);
    return buf;
}

static int eval(const char* name, const char* data)
{
    char path[160];
    snprintf(path, sizeof path, DIR "%s.knx", name);
    kernix_model* m = kernix_model_load(path);
    if (m == NULL) return 1;
    snprintf(path, sizeof path, DIR "%s_x.bin", data);
    float_tensor* x = load_tensor(path);
    snprintf(path, sizeof path, DIR "%s_y.bin", data);
    float_tensor* want = load_tensor(path);
    snprintf(path, sizeof path, DIR "%s_labels.bin", data);
    float_tensor* labels = load_tensor(path);
    const int T = m->seq_len, n = want->row, cls = want->col;

    const float_tensor* y = kernix_model_forward(m, x);
    if (y == NULL || y->row != n || y->col != cls) { fprintf(stderr, "forward 실패 또는 출력 모양 불일치\n"); return 1; }
    int agree = 0, hit = 0;
    for (int i = 0; i < n; i++) {
        int a = 0, w = 0;
        for (int j = 1; j < cls; j++) {
            if (y->tensor[i * cls + j] > y->tensor[i * cls + a]) a = j;
            if (want->tensor[i * cls + j] > want->tensor[i * cls + w]) w = j;
        }
        agree += a == w;
        hit += a == (int)labels->tensor[i];
    }
    printf("max_abs_err %.3e\nargmax_agree %.9f\naccuracy %.9f\n", max_abs(y, want), (double)agree / n, (double)hit / n);

    double t[200];
    double c0 = cpu_sec(), st = now();
    for (int i = 0; i < 5; i++) { st = now(); kernix_model_forward(m, x); t[i] = (now() - st) * 1e3; }
    double cpu_all = (cpu_sec() - c0) * 1e3 / 5;
    qsort(t, 5, sizeof(double), cmp_double);
    printf("all %.4f %.4f\n", t[2], cpu_all);

    /* 표본 1개씩: 웹캠 추론과 같은 조건. 표본마다 다른 입력 */
    float_tensor one = { T, x->col, x->tensor };
    kernix_model_forward(m, &one); /* 배치 1 버퍼 할당 + 워밍업 */
    c0 = cpu_sec();
    for (int i = 0; i < 200; i++) {
        one.tensor = x->tensor + (size_t)(i % n) * T * x->col;
        st = now();
        if (kernix_model_forward(m, &one) == NULL) return 1;
        t[i] = (now() - st) * 1e3;
    }
    double cpu_one = (cpu_sec() - c0) * 1e3 / 200;
    qsort(t, 200, sizeof(double), cmp_double);
    printf("single %.5f %.5f\n", (t[99] + t[100]) / 2.0, cpu_one);

    struct rusage u;
    getrusage(RUSAGE_SELF, &u);
    printf("maxrss_mb %.1f\n", (double)u.ru_maxrss / (1 << 20));
    float_free(x); float_free(want); float_free(labels);
    kernix_model_free(m);
    return 0;
}

/* int8 커널 vs Accelerate: 행 수별 중앙값 시간 (KERNIX_INT8_MAX_ROWS를 정하는 측정) */
static int qbench(void)
{
    const int shapes[][2] = { { 64, 256 }, { 128, 512 }, { 784, 1024 } }, rows_list[] = { 1, 2, 4, 8, 12, 16, 24, 32 };
    printf("int8 sdot vs cblas_sgemm, 중앙값 ns (int8 / float), K×N 별\n%-6s", "rows");
    for (int s = 0; s < 3; s++) printf(" | %4d×%-4d     int8      float", shapes[s][0], shapes[s][1]);
    printf("\n");
    for (int ri = 0; ri < 8; ri++) {
        const int rows = rows_list[ri];
        printf("%-6d", rows);
        for (int s = 0; s < 3; s++) {
            const int k = shapes[s][0], n = shapes[s][1];
            int8_t* q = malloc((size_t)k * n);
            float *scale = malloc(sizeof(float) * n), *w = malloc(sizeof(float) * k * n);
            float *x = malloc(sizeof(float) * rows * k), *y = malloc(sizeof(float) * rows * n);
            for (size_t i = 0; i < (size_t)k * n; i++) { q[i] = (int8_t)((int)(qrand() % 255) - 127); w[i] = q[i] * 0.01f; }
            for (int i = 0; i < n; i++) scale[i] = 0.01f;
            for (int i = 0; i < rows * k; i++) x[i] = (float)(qrand() % 2001) * 0.001f - 1.0f;
            kernix_qweight* qw = kernix_qweight_pack(q, scale, k, n);
            const int reps = (int)(4e6 / ((double)rows * k * n / 16 + 200)) + 5;
            double tq[21], tf[21];
            for (int t = 0; t < 21; t++) {
                double st = now();
                for (int r = 0; r < reps; r++) kernix_qmatmul(x, rows, qw, NULL, 0, y);
                tq[t] = (now() - st) / reps * 1e9;
                st = now();
                for (int r = 0; r < reps; r++)
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, n, k, 1.0f, x, k, w, n, 0.0f, y, n);
                tf[t] = (now() - st) / reps * 1e9;
            }
            qsort(tq, 21, sizeof(double), cmp_double);
            qsort(tf, 21, sizeof(double), cmp_double);
            printf(" | %9.0f %9.0f %s", tq[10], tf[10], tq[10] < tf[10] ? "int8" : "float");
            kernix_qweight_free(qw);
            free(q); free(scale); free(w); free(x); free(y);
        }
        printf("\n");
    }
    return 0;
}

/* 현재 메모리 사용량 (MB): 활성 상태 보기의 "메모리"와 같은 phys_footprint. resident_size와 달리 free()가
 * 재사용 표시만 한 페이지는 빠지므로, 해제(trim)가 실제로 운영체제에 돌려준 양이 보임 */
static double rss_mb(void)
{
    task_vm_info_data_t info;
    mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &n) != KERN_SUCCESS) return -1;
    return (double)info.phys_footprint / (1 << 20);
}

/* 전체 검증 데이터를 한 번에 넣을 때: 상한별 시간(5회 중앙값), forward로 늘어난 최대 메모리, trim 전후 상주 메모리.
 * "bench"는 MLP 벤치 모델에 4096행 합성 입력 (기대 출력 없이 상한 0과 비교) */
static int mem(const char* name, double limit_mb)
{
    char path[160];
    snprintf(path, sizeof path, DIR "%s.knx", name);
    kernix_model* m = kernix_model_load(path);
    if (m == NULL) return 1;
    float_tensor *x, *want = NULL;
    if (strcmp(name, "bench") == 0) {
        x = float_zeros(4096, m->in_dim);
        for (size_t i = 0; i < (size_t)x->row * x->col; i++) x->tensor[i] = (float)((int)(i % 200) - 100) * 0.01f;
    } else {
        char dn[160];
        snprintf(path, sizeof path, DIR "%s_x.bin", data_name(name, dn, sizeof dn));
        x = load_tensor(path);
        snprintf(path, sizeof path, DIR "%s_y.bin", dn);
        want = load_tensor(path);
    }
    if (x == NULL) return 1;
    const double base = rss_mb(); /* 모델 + 입력 + 기대 출력까지 올린 상태 */
    kernix_model_set_workspace_limit(m, (size_t)(limit_mb * (1 << 20)));
    double t[5];
    const float_tensor* y = NULL;
    for (int i = 0; i < 5; i++) {
        double st = now();
        if ((y = kernix_model_forward(m, x)) == NULL) return 1;
        t[i] = (now() - st) * 1e3;
    }
    qsort(t, 5, sizeof(double), cmp_double);
    struct rusage u;
    getrusage(RUSAGE_SELF, &u);
    const double peak = (double)u.ru_maxrss / (1 << 20), after = rss_mb();
    /* 런타임이 실제로 잡은 작업 버퍼 (정확한 값). 프로세스 메모리 차이는 malloc 캐시 때문에 들쭉날쭉함 */
    size_t deq = 0; /* int8 단계의 float 사본 (COUNT 모드는 세지 않음) */
    for (int i = 0; i < m->n_steps; i++)
        if (m->steps[i].deq != NULL) deq += (size_t)m->steps[i].deq->row * (size_t)m->steps[i].deq->col;
    const double ws = (double)(kernix_model_shape(m, m->batch, KERNIX_SHAPE_COUNT) + deq +
                               (m->out_all ? (size_t)m->out_all->row * (size_t)m->out_all->col : 0)) * sizeof(float) / (1 << 20);
    printf("samples %d\nchunk %d\nms %.3f\nws_mb %.2f\nbase_mb %.1f\nmaxrss_mb %.1f\nafter_mb %.1f\n",
           x->row / m->seq_len, m->batch / m->seq_len, t[2], ws, base, peak, after);
    if (want != NULL) printf("max_abs_err %.3e\n", max_abs(y, want));
    kernix_model_trim(m);
    usleep(1000000); /* Metal 드라이버는 해제된 버퍼의 페이지를 비동기로(1초 안쪽) 반환 */
    printf("trim_mb %.1f\n", rss_mb());
    float_free(x); float_free(want);
    kernix_model_free(m);
    return 0;
}

int main(int argc, char** argv)
{
    if (argc == 4 && strcmp(argv[1], "--mem") == 0) return mem(argv[2], atof(argv[3]));
    if (argc == 2 && strcmp(argv[1], "--qbench") == 0) return qbench();
    if (argc == 3 && strcmp(argv[1], "--bench") == 0) return bench(argv[2]);
    char dn[160];
    if ((argc == 3 || argc == 4) && strcmp(argv[1], "--eval") == 0) return eval(argv[2], argc == 4 ? argv[3] : data_name(argv[2], dn, sizeof dn));
    printf("=== MLP 런타임 검증 ===\n");
    check_load();
    check_forward();
    check_lstm();
    check_chunk();
    check_qmatmul();
    check_v2_load();
    check_v2_forward();
    return fail;
}
