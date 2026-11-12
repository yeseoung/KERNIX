#ifndef KERNIX_H
#define KERNIX_H

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stddef.h>
#include <math.h>
#if defined(__APPLE__)
#include <sys/mman.h>
#include <malloc/malloc.h>
#endif

/* 이 크기 이상의 블록은 해제할 때 페이지를 운영체제에 바로 반환 (macOS malloc은 큰 블록을 캐시에 남겨 메모리가 안 줄어듦) */
#define KERNIX_RELEASE_MIN_BYTES ((size_t)1 << 20)

#if defined(__APPLE__) && (defined(KERNIX_USE_ACCELERATE) || defined(__ACCELERATE__))
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#define KERNIX_ACCELERATE_ENABLED 1
#endif

/* Apple Silicon 페이지 크기(16KB), 4KB 페이지의 배수라 Intel/Linux에서도 유효 */
#define KERNIX_PAGE 16384

#if defined(KERNIX_USE_METAL)
/* kernix.h를 포함하는 모든 소스에서 정의해야 함: 아니면 float_free가 Metal 메모리를 free()로 해제 */
/* 이 크기 이상의 float 텐서는 Metal 공유 버퍼로 할당: GPU 매핑을 할당 시 한 번만 수행 */
#define KERNIX_METAL_MIN_BYTES (1 << 16)
void* kernix_metal_alloc(size_t bytes); /* 실패 시 NULL */
int kernix_metal_free(void* p);         /* Metal 할당이면 해제 후 1, 아니면 0 */
void kernix_metal_release_cache(void);  /* 해제됐지만 재사용 캐시에 남은 Metal 버퍼를 운영체제에 반환 */
#endif

typedef struct{
    int row;
    int col;
    float* tensor;
}float_tensor;

typedef struct
{
    int row;
    int col;
    int* tensor;
}int_tensor;

typedef struct
{
    int row;
    int col;
    uint8_t* tensor;

}uint8_tensor;

/* ── 스레드 풀 (src/kernix_runtime.c에 구현, 같이 빌드해야 함) ──
 * 작업자 7개 + 호출 스레드 = 8개. 작업을 n개 조각으로 나눠 주면 각 스레드가 남은 조각을 하나씩 가져가므로
 * 빠른 코어가 더 많이 처리한다. n이 1 이하면 작업자를 깨우지 않고 호출 스레드에서 바로 실행. 실패 시 -1 */
#define KERNIX_POOL_WORKERS 7
typedef void (*kernix_chunk_fn)(void* ctx, size_t chunk);
int kernix_pool_run(size_t n_chunks, kernix_chunk_fn fn, void* ctx);
int kernix_thread_pool_init(void);      /* 미리 만들어 두기 (선택: kernix_pool_run이 처음 부를 때 만듦) */
void kernix_thread_pool_shutdown(void); /* 작업자 종료. 이후 kernix_pool_run은 다시 만든다 */

/* 합산 조각: 64K 원소(256KB). 조각마다 결과를 따로 두고 순서대로 더해 실행마다 결과가 같다 */
#define KERNIX_SUM_CHUNK ((size_t)1 << 16)
typedef struct { const float* data; size_t n; double* part; } kernix_sum_ctx;

static void kernix_sum_chunk(void* p, size_t c)
{
    const kernix_sum_ctx* x = (const kernix_sum_ctx*)p;
    size_t i = c * KERNIX_SUM_CHUNK;
    const size_t end = x->n - i < KERNIX_SUM_CHUNK ? x->n : i + KERNIX_SUM_CHUNK;
    float sum[8] = {0.0f};
    for (; end - i >= 8; i += 8) {
        for (size_t j = 0; j < 8; j++) {
            sum[j] += x->data[i + j];
        }
    }
    for (; i < end; i++) {
        sum[0] += x->data[i];
    }
    double total = 0.0;
    for (size_t j = 0; j < 8; j++) {
        total += sum[j];
    }
    x->part[c] = total;
}

/* 행렬 곱셈 조각: 행 묶음 하나. 약 64조각으로 나누고, 곱셈이 2²⁰번 미만이면 한 조각(직렬) */
#define KERNIX_MM_PIECES 64
#define KERNIX_MM_PAR_MIN ((size_t)1 << 20)
typedef struct { const void* a; const void* b; void* c; int k, n, rows, per; } kernix_mm_ctx;

static inline size_t kernix_mm_split(kernix_mm_ctx* x)
{
    size_t work = (size_t)x->rows * (size_t)x->k * (size_t)x->n;
    x->per = work < KERNIX_MM_PAR_MIN ? x->rows : (x->rows + KERNIX_MM_PIECES - 1) / KERNIX_MM_PIECES;
    if (x->per < 1) x->per = 1;
    return x->rows == 0 ? 0 : ((size_t)x->rows + x->per - 1) / x->per;
}

/* i‑k‑j 순서: B를 행 단위로 접근하여 캐시 효율 향상 */
#define KERNIX_MM_CHUNK(name, T)                                                \
static void name(void* p, size_t ch)                                           \
{                                                                              \
    const kernix_mm_ctx* x = (const kernix_mm_ctx*)p;                          \
    const T* a = (const T*)x->a; const T* b = (const T*)x->b; T* c = (T*)x->c; \
    int r0 = (int)ch * x->per, r1 = r0 + x->per < x->rows ? r0 + x->per : x->rows; \
    for (int i = r0; i < r1; i++) {                                            \
        for (int k = 0; k < x->k; k++) {                                       \
            T a_ik = a[(size_t)i * x->k + k];                                  \
            for (int j = 0; j < x->n; j++) {                                   \
                c[(size_t)i * x->n + j] += a_ik * b[(size_t)k * x->n + j];     \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}
KERNIX_MM_CHUNK(kernix_float_mm_chunk, float)
KERNIX_MM_CHUNK(kernix_int_mm_chunk, int)

static inline int_tensor* int_zeros(int rows, int cols)
{
    if (rows < 0 || cols < 0) return NULL;
    int_tensor* p = malloc(sizeof(int_tensor));
    if (p == NULL) return NULL;

    p->row = rows;
    p->col = cols;
    size_t count = (size_t)rows * (size_t)cols;
    p->tensor = malloc(sizeof(int) * (count ? count : 1)); /* 0x0도 NULL 아닌 포인터 */
    if (p->tensor == NULL) {
        free(p);
        return NULL;
    }

    memset(p->tensor, 0, sizeof(int) * count);
    return p;
}

/* zero=0이면 값 초기화 생략: 결과를 전부 덮어쓰는 연산(행렬 곱 출력)용 */
static inline float_tensor* kernix_float_new(int rows, int cols, int zero)
{
    if (rows < 0 || cols < 0) return NULL;
    float_tensor* p = malloc(sizeof(float_tensor));
    if (p == NULL) return NULL;
    
    p->row = rows;
    p->col = cols;
    size_t count = (size_t)rows * (size_t)cols;
    /* 페이지 정렬 + 페이지 단위 크기: Metal이 복사 없이(no-copy) 이 메모리를 GPU 버퍼로 사용 */
    size_t bytes = (sizeof(float) * count + KERNIX_PAGE - 1) / KERNIX_PAGE * KERNIX_PAGE;
    void* mem = NULL;
#if defined(KERNIX_USE_METAL)
    if (bytes >= KERNIX_METAL_MIN_BYTES) mem = kernix_metal_alloc(bytes);
#endif
    if (mem == NULL) {
        if (posix_memalign(&mem, KERNIX_PAGE, bytes ? bytes : KERNIX_PAGE) != 0) mem = NULL;
#if defined(__APPLE__)
        /* float_free가 반환 표시한 블록을 malloc이 다시 줬으면 사용 중으로 되돌림 (메모리 사용량 집계가 맞게) */
        else if (bytes >= KERNIX_RELEASE_MIN_BYTES) madvise(mem, bytes, MADV_FREE_REUSE);
#endif
    }
    p->tensor = mem;
    if (p->tensor == NULL) {
        free(p);
        return NULL;
    }

    if (zero) memset(p->tensor, 0, sizeof(float) * count);
    return p;
}

static inline float_tensor* float_zeros(int rows, int cols)
{
    return kernix_float_new(rows, cols, 1);
}

static inline uint8_tensor* uint8_zeros(int rows, int cols)
{
    if (rows < 0 || cols < 0) return NULL;
    uint8_tensor* p = malloc(sizeof(uint8_tensor));
    if (p == NULL) return NULL;

    p->row = rows;
    p->col = cols;
    size_t count = (size_t)rows * (size_t)cols;
    p->tensor = malloc(sizeof(uint8_t) * (count ? count : 1)); /* 0x0도 NULL 아닌 포인터 */
    if (p->tensor == NULL) {
        free(p);
        return NULL;
    }

    memset(p->tensor, 0, sizeof(uint8_t) * count);
    return p;
}

static inline void int_free(int_tensor* t){
    if(t == NULL){
        return;
    }
    free(t->tensor);
    free(t);
}
static inline void float_free(float_tensor* t){
    if(t == NULL){
        return;
    }
#if defined(KERNIX_USE_METAL)
    if (!kernix_metal_free(t->tensor))
#endif
    {
#if defined(__APPLE__)
        /* 큰 블록은 malloc 캐시에 남아도 페이지는 지금 반환 (실제 크기는 malloc_size: row는 뷰로 줄어 있을 수 있음) */
        /* 페이지 단위로 내림: 커널은 길이를 올림하므로, 배수가 아니면 블록 뒤의 다른 할당 페이지까지 반환 표시될 수 있음.
         * KERNIX_PAGE(16KB)는 4KB·16KB 페이지 모두의 배수. 시작 주소는 posix_memalign(KERNIX_PAGE)이라 이미 정렬됨 */
        size_t n = malloc_size(t->tensor) / KERNIX_PAGE * KERNIX_PAGE;
        if (n >= KERNIX_RELEASE_MIN_BYTES) madvise(t->tensor, n, MADV_FREE_REUSABLE);
#endif
        free(t->tensor);
    }
    free(t);
}
static inline void uint8_free(uint8_tensor* t){
    if(t == NULL){
        return;
    }
    free(t->tensor);
    free(t);
}

static inline void show_ints_impl(int_tensor* tensor,const char *name) {

    printf("%s : [\n", name);
    int rows = tensor->row;
    int cols = tensor->col;

    for (int i = 0; i < rows; i++) {
        printf("  { ");

        for (int j = 0; j < cols; j++) {
            printf("%3d", tensor->tensor[i * cols + j]);

            if (j < cols - 1) {
                printf(" , ");
            }
        }

        printf(" }\n");
    }

    printf("]\n");
}
static inline void show_floats_impl(float_tensor* tensor,const char *name) {

    printf("%s : [\n", name);
    int rows = tensor->row;
    int cols = tensor->col;

    for (int i = 0; i < rows; i++) {
        printf("  { ");

        for (int j = 0; j < cols; j++) {
            printf("%7.2f", tensor->tensor[i * cols + j]);

            if (j < cols - 1) {
                printf(" , ");
            }
        }

        printf(" }\n");
    }

    printf("]\n");
}

static inline void show_in8ts_impl(
    uint8_tensor *tensor,
    const char *name,
    int bits
) {
    printf("%s : [\n", name);

    int rows = tensor->row;
    int cols = tensor->col;

    for (int row = 0; row < rows; row++) {
        printf("  { ");

        for (int col = 0; col < cols; col++) {
            uint8_t value = tensor->tensor[row * cols + col];

            if (!bits) {
                printf("%3d", value);
            } else {
                for (int bit = 7; bit >= 0; bit--) {
                    printf("%d", (value >> bit) & 1);
                }
            }

            if (col < cols - 1) {
                printf(" , ");
            }
        }

        printf(" }\n");
    }

    printf("]\n");
}


static inline int int_dot(const int_tensor* a, const int_tensor* b) {
    if (a == NULL || b == NULL || a->tensor == NULL || b->tensor == NULL) {
        return -1;
    }
    size_t a_size = (size_t)a->row * (size_t)a->col;
    size_t b_size = (size_t)b->row * (size_t)b->col;

    if (a_size != b_size) {
        /* shortcut: -1은 정상 결과와 구분 불가, 호출자가 오류 판별이 필요하면 out-param으로 변경 */
        fprintf(stderr, "내적 크기 불일치: %zu vs %zu\n", a_size, b_size);
        return -1;
    }

    int sum = 0;
    for (size_t i = 0; i < a_size; i++) {
        sum += a->tensor[i] * b->tensor[i];
    }
    return sum;
}

static inline float float_dot(const float_tensor* a, const float_tensor* b) {
    if (a == NULL || b == NULL || a->tensor == NULL || b->tensor == NULL) {
        return NAN;
    }
    size_t a_size = (size_t)a->row * (size_t)a->col;
    size_t b_size = (size_t)b->row * (size_t)b->col;

    if (a_size != b_size) {
        fprintf(stderr, "내적 크기 불일치: %zu vs %zu\n", a_size, b_size);
        return NAN;
    }

    float sum = 0.0f;
    for (size_t i = 0; i < a_size; i++) {
        sum += a->tensor[i] * b->tensor[i];
    }
    return sum;
}

static inline int uint8_dot(const uint8_tensor* a, const uint8_tensor* b) {
    if (a == NULL || b == NULL || a->tensor == NULL || b->tensor == NULL) {
        return -1;
    }
    size_t a_size = (size_t)a->row * (size_t)a->col;
    size_t b_size = (size_t)b->row * (size_t)b->col;

    if (a_size != b_size) {
        /* shortcut: -1은 정상 결과와 구분 불가, 호출자가 오류 판별이 필요하면 out-param으로 변경 */
        fprintf(stderr, "내적 크기 불일치: %zu vs %zu\n", a_size, b_size);
        return -1;
    }

    int sum = 0;
    for (size_t i = 0; i < a_size; i++) {
        sum += (int)a->tensor[i] * (int)b->tensor[i];
    }
    return sum;
}

#if defined(KERNIX_ACCELERATE_ENABLED)
/* ══ PyTorch 핵심 연산 (Apple 네이티브) ══
 * Accelerate의 vDSP/vForce(SIMD)가 각 구간을 계산하고, GCD가 구간을 코어에 분배.
 * 구간 = 32KB라 다단계 연산(sigmoid, softmax 등)의 중간값이 캐시에 머물러 메모리를 한 번만 왕복 */
#include <dispatch/dispatch.h>

#define KERNIX_CHUNK 8192          /* float 8192개 = 32KB */
#define KERNIX_PAR_MIN (1 << 16)   /* 이보다 작으면 GCD 분배 비용(~수 µs)이 더 큼 */

typedef void (^kernix_range_fn)(size_t start, size_t len);

/* [0, units)를 구간으로 나눠 fn 실행. unit_cost = 단위(원소 또는 행)당 원소 수 */
static inline void kernix_par(size_t units, size_t unit_cost, kernix_range_fn fn)
{
    size_t per = unit_cost >= KERNIX_CHUNK ? 1 : KERNIX_CHUNK / (unit_cost ? unit_cost : 1);
    size_t chunks = (units + per - 1) / per;
    if (units * unit_cost < KERNIX_PAR_MIN) {
        for (size_t c = 0; c < chunks; c++) {
            size_t s = c * per;
            fn(s, units - s < per ? units - s : per);
        }
        return;
    }
    dispatch_apply(chunks, DISPATCH_APPLY_AUTO, ^(size_t c) {
        size_t s = c * per;
        fn(s, units - s < per ? units - s : per);
    });
}

#define KERNIX_UNPAREN(...) __VA_ARGS__

/* 큰 텐서 원소 연산을 Metal로 보내는 연결점. 번호는 src/kernix_metal.m 커널의 switch와 일치 */
enum {
    KERNIX_GPU_NEG, KERNIX_GPU_ABS, KERNIX_GPU_EXP, KERNIX_GPU_LOG, KERNIX_GPU_SQRT, KERNIX_GPU_TANH,
    KERNIX_GPU_RELU, KERNIX_GPU_SIGMOID, KERNIX_GPU_SILU, KERNIX_GPU_GELU,
    KERNIX_GPU_ADD_SCALAR, KERNIX_GPU_MUL_SCALAR, KERNIX_GPU_CLAMP,
    KERNIX_GPU_ADD, KERNIX_GPU_SUB, KERNIX_GPU_MUL, KERNIX_GPU_DIV
};
#if defined(KERNIX_USE_METAL)
/* 앞쪽 원소를 GPU로 계산하고 처리한 원소 수를 반환(0 = GPU 미사용). b: 같은 모양(bcols=0) 또는 1×bcols */
size_t kernix_metal_map(int op, const float* x, const float* b, size_t bcols, float* y, size_t n, float p0, float p1);
#define KERNIX_GPU_MAP kernix_metal_map
/* 행 단위 연산(mode 0 softmax, 1 log_softmax, 2 layer_norm)과 전치를 GPU로. 처리했으면 1 */
int kernix_metal_rows(int mode, const float* x, const float* g, const float* b, float eps,
                      float* y, size_t rows, size_t cols);
int kernix_metal_transpose(const float* x, float* y, size_t rows, size_t cols);
#define KERNIX_GPU_ROWS(mode, g, b, eps) kernix_metal_rows(mode, in, g, b, eps, out, rows, cols)
#else
#define KERNIX_GPU_ROWS(mode, g, b, eps) ((void)(g), (void)(b), (void)(eps), 0)
static inline size_t kernix_gpu_none(int op, const float* x, const float* b, size_t bcols, float* y, size_t n,
                                     float p0, float p1)
{
    (void)op; (void)x; (void)b; (void)bcols; (void)y; (void)n; (void)p0; (void)p1;
    return 0;
}
#define KERNIX_GPU_MAP kernix_gpu_none
#endif

/* 원소별 연산 생성. 큰 텐서는 GPU(gpu_op, 인자 gp)가 먼저 처리하고 남은 꼬리는 CPU 본문이 처리.
 * 본문에서 x(입력), o(출력), len/n(구간 길이)을 사용 */
#define KERNIX_MAP(name, gpu_op, gp, params, ...)                              \
static inline float_tensor* name(const float_tensor* a KERNIX_UNPAREN params)  \
{                                                                              \
    if (a == NULL || a->tensor == NULL) return NULL;                           \
    float_tensor* y = kernix_float_new(a->row, a->col, 0);                     \
    if (y == NULL) return NULL;                                                \
    const float* in = a->tensor;                                               \
    float* out = y->tensor;                                                    \
    const size_t count = (size_t)a->row * a->col;                              \
    const size_t done = KERNIX_GPU_MAP(gpu_op, in, NULL, 0, out, count, KERNIX_UNPAREN gp); \
    kernix_par(count - done, 1, ^(size_t s, size_t len) {                      \
        const float* x = in + done + s;                                        \
        float* o = out + done + s;                                             \
        const int n = (int)len;                                                \
        (void)x; (void)n;                                                      \
        __VA_ARGS__;                                                           \
    });                                                                        \
    return y;                                                                  \
}

KERNIX_MAP(floats_neg, KERNIX_GPU_NEG, (0, 0), (), vDSP_vneg(x, 1, o, 1, len))
KERNIX_MAP(floats_abs, KERNIX_GPU_ABS, (0, 0), (), vDSP_vabs(x, 1, o, 1, len))
KERNIX_MAP(floats_exp, KERNIX_GPU_EXP, (0, 0), (), vvexpf(o, x, &n))
KERNIX_MAP(floats_log, KERNIX_GPU_LOG, (0, 0), (), vvlogf(o, x, &n))
KERNIX_MAP(floats_sqrt, KERNIX_GPU_SQRT, (0, 0), (), vvsqrtf(o, x, &n))
KERNIX_MAP(floats_tanh, KERNIX_GPU_TANH, (0, 0), (), vvtanhf(o, x, &n))
KERNIX_MAP(floats_relu, KERNIX_GPU_RELU, (0, 0), (), const float z = 0.0f; vDSP_vthr(x, 1, &z, o, 1, len))
/* sigmoid(x) = 1 / (1 + e^-x) */
KERNIX_MAP(floats_sigmoid, KERNIX_GPU_SIGMOID, (0, 0), (),
    const float one = 1.0f;
    vDSP_vneg(x, 1, o, 1, len); vvexpf(o, o, &n);
    vDSP_vsadd(o, 1, &one, o, 1, len); vvrecf(o, o, &n))
/* silu(x) = x * sigmoid(x) */
KERNIX_MAP(floats_silu, KERNIX_GPU_SILU, (0, 0), (),
    const float one = 1.0f;
    vDSP_vneg(x, 1, o, 1, len); vvexpf(o, o, &n);
    vDSP_vsadd(o, 1, &one, o, 1, len); vvrecf(o, o, &n);
    vDSP_vmul(x, 1, o, 1, o, 1, len))
/* gelu(x) = x/2 * (1 + erf(x/√2)), torch 기본(정확한 erf) 방식.
 * erf는 Abramowitz-Stegun 7.1.26 (절대 오차 1.5e-7, float32 정밀도 수준) */
KERNIX_MAP(floats_gelu, KERNIX_GPU_GELU, (0, 0), (),
    float t[KERNIX_CHUNK];
    for (size_t i = 0; i < len; i++) { float z = x[i] * 0.70710678f; t[i] = -z * z; }
    vvexpf(t, t, &n);
    for (size_t i = 0; i < len; i++) {
        float z = fabsf(x[i]) * 0.70710678f;
        float k = 1.0f / (1.0f + 0.3275911f * z);
        float p = k * (0.254829592f + k * (-0.284496736f + k * (1.421413741f
                + k * (-1.453152027f + k * 1.061405429f))));
        float e = 1.0f - p * t[i];
        o[i] = 0.5f * x[i] * (1.0f + (x[i] < 0.0f ? -e : e));
    })
KERNIX_MAP(floats_add_scalar, KERNIX_GPU_ADD_SCALAR, (v, 0), (, float v), vDSP_vsadd(x, 1, &v, o, 1, len))
KERNIX_MAP(floats_mul_scalar, KERNIX_GPU_MUL_SCALAR, (v, 0), (, float v), vDSP_vsmul(x, 1, &v, o, 1, len))
KERNIX_MAP(floats_clamp, KERNIX_GPU_CLAMP, (lo, hi), (, float lo, float hi), vDSP_vclip(x, 1, &lo, &hi, o, 1, len))

typedef enum { KERNIX_ADD, KERNIX_SUB, KERNIX_MUL, KERNIX_DIV } kernix_binop;

static inline void kernix_binop_run(kernix_binop op, const float* a, const float* b, float* o, size_t n)
{
    switch (op) {
    case KERNIX_ADD: vDSP_vadd(a, 1, b, 1, o, 1, n); break;
    case KERNIX_SUB: vDSP_vsub(b, 1, a, 1, o, 1, n); break; /* vDSP 인자 순서: 두 번째 - 첫 번째 */
    case KERNIX_MUL: vDSP_vmul(a, 1, b, 1, o, 1, n); break;
    case KERNIX_DIV: vDSP_vdiv(b, 1, a, 1, o, 1, n); break; /* vDSP 인자 순서: 두 번째 / 첫 번째 */
    }
}

/* b는 a와 같은 모양, 또는 1×cols 행 벡터(모든 행에 브로드캐스트, 예: 편향) */
static inline float_tensor* kernix_binary(const float_tensor* a, const float_tensor* b, kernix_binop op)
{
    if (a == NULL || b == NULL || a->tensor == NULL || b->tensor == NULL) return NULL;
    int same = a->row == b->row && a->col == b->col;
    if (!same && !(b->row == 1 && b->col == a->col)) {
        fprintf(stderr, "원소 연산 차원 불일치: A(%d×%d) vs B(%d×%d)\n", a->row, a->col, b->row, b->col);
        return NULL;
    }
    float_tensor* y = kernix_float_new(a->row, a->col, 0);
    if (y == NULL) return NULL;
    const float* pa = a->tensor;
    const float* pb = b->tensor;
    float* po = y->tensor;
    size_t cols = (size_t)a->col;
    size_t count = (size_t)a->row * cols;
    /* GPU는 브로드캐스트 시 cols가 4의 배수일 때만 쓰므로, 그 경우 전체(done == count)를 처리 */
    size_t done = KERNIX_GPU_MAP(KERNIX_GPU_ADD + (int)op, pa, pb, same ? 0 : cols, po, count, 0, 0);
    if (same) {
        kernix_par(count - done, 1, ^(size_t s, size_t len) {
            kernix_binop_run(op, pa + done + s, pb + done + s, po + done + s, len);
        });
    } else if (done < count) {
        kernix_par((size_t)a->row, cols, ^(size_t r0, size_t nr) {
            for (size_t r = r0; r < r0 + nr; r++) kernix_binop_run(op, pa + r * cols, pb, po + r * cols, cols);
        });
    }
    return y;
}

static inline float_tensor* floats_sub(const float_tensor* a, const float_tensor* b) { return kernix_binary(a, b, KERNIX_SUB); }
static inline float_tensor* floats_mul(const float_tensor* a, const float_tensor* b) { return kernix_binary(a, b, KERNIX_MUL); }
static inline float_tensor* floats_div(const float_tensor* a, const float_tensor* b) { return kernix_binary(a, b, KERNIX_DIV); }

typedef enum { KERNIX_SUM, KERNIX_MAX, KERNIX_MIN } kernix_reduce_op;

/* 전체 축소. NULL 또는 (max/min의) 빈 텐서는 NAN */
static inline float kernix_reduce(const float_tensor* a, kernix_reduce_op op)
{
    if (a == NULL || a->tensor == NULL) return NAN;
    size_t count = (size_t)a->row * a->col;
    if (count == 0) return op == KERNIX_SUM ? 0.0f : NAN;
    size_t chunks = (count + KERNIX_CHUNK - 1) / KERNIX_CHUNK;
    float* part = malloc(chunks * sizeof(float));
    if (part == NULL) return NAN;
    const float* in = a->tensor;
    kernix_par(count, 1, ^(size_t s, size_t len) {
        float* r = &part[s / KERNIX_CHUNK];
        if (op == KERNIX_SUM) vDSP_sve(in + s, 1, r, len);
        else if (op == KERNIX_MAX) vDSP_maxv(in + s, 1, r, len);
        else vDSP_minv(in + s, 1, r, len);
    });
    double acc = part[0];
    for (size_t c = 1; c < chunks; c++) {
        if (op == KERNIX_SUM) acc += part[c]; /* 구간 합은 double로 누적해 오차 억제 */
        else if (op == KERNIX_MAX) acc = part[c] > acc ? part[c] : acc;
        else acc = part[c] < acc ? part[c] : acc;
    }
    free(part);
    return (float)acc;
}

static inline float floats_sum(const float_tensor* a) { return kernix_reduce(a, KERNIX_SUM); }
static inline float floats_max(const float_tensor* a) { return kernix_reduce(a, KERNIX_MAX); }
static inline float floats_min(const float_tensor* a) { return kernix_reduce(a, KERNIX_MIN); }
static inline float floats_mean(const float_tensor* a)
{
    if (a == NULL || a->tensor == NULL || a->row == 0 || a->col == 0) return NAN;
    return floats_sum(a) / ((float)a->row * (float)a->col);
}

/* dim=0: 열 합(1×cols), dim=1: 행 합(rows×1). torch.sum(x, dim) */
static inline float_tensor* floats_sum_dim(const float_tensor* a, int dim)
{
    if (a == NULL || a->tensor == NULL || (dim != 0 && dim != 1)) return NULL;
    size_t rows = (size_t)a->row, cols = (size_t)a->col;
    float_tensor* y = dim == 0 ? float_zeros(1, a->col) : float_zeros(a->row, 1);
    if (y == NULL) return NULL;
    const float* in = a->tensor;
    float* out = y->tensor;
    if (dim == 1) {
        kernix_par(rows, cols, ^(size_t r0, size_t nr) {
            for (size_t r = r0; r < r0 + nr; r++) vDSP_sve(in + r * cols, 1, &out[r], cols);
        });
    } else {
        /* 행 블록마다 부분 열합(연속 접근)을 만든 뒤 합침 */
        size_t blocks = rows * cols < KERNIX_PAR_MIN ? 1 : (rows < 8 ? rows : 8);
        float* part = calloc(blocks * cols, sizeof(float));
        if (part == NULL) { float_free(y); return NULL; }
        dispatch_apply(blocks, DISPATCH_APPLY_AUTO, ^(size_t k) {
            float* acc = part + k * cols;
            for (size_t r = k * rows / blocks; r < (k + 1) * rows / blocks; r++)
                vDSP_vadd(acc, 1, in + r * cols, 1, acc, 1, cols);
        });
        for (size_t k = 0; k < blocks; k++) vDSP_vadd(out, 1, part + k * cols, 1, out, 1, cols);
        free(part);
    }
    return y;
}

/* 행별 최댓값 위치(rows×1). torch.argmax(x, dim=1) */
static inline int_tensor* floats_argmax_rows(const float_tensor* a)
{
    if (a == NULL || a->tensor == NULL || a->col == 0) return NULL;
    int_tensor* y = int_zeros(a->row, 1);
    if (y == NULL) return NULL;
    size_t cols = (size_t)a->col;
    const float* in = a->tensor;
    int* out = y->tensor;
    kernix_par((size_t)a->row, cols, ^(size_t r0, size_t nr) {
        for (size_t r = r0; r < r0 + nr; r++) {
            float v;
            vDSP_Length idx;
            vDSP_maxvi(in + r * cols, 1, &v, &idx, cols);
            out[r] = (int)idx;
        }
    });
    return y;
}

/* 행 단위 연산 생성: name_into(주어진 버퍼에 씀)와 name(할당 후 _into 호출).
 * 큰 텐서는 gpu 식이 먼저 시도. in == out(제자리)이면 GPU를 쓰지 않음: Metal 커널은 같은 버퍼 읽기/쓰기를 보장 안 함.
 * args는 name이 _into로 넘길 추가 인자. 본문에서 x(입력 행), o(출력 행), cols, n을 사용 */
#define KERNIX_ROWS(name, gpu, params, args, ...)                              \
static inline void name##_into(const float* in, float* out, size_t rows,       \
                               size_t cols KERNIX_UNPAREN params)              \
{                                                                              \
    if (in != out && (gpu)) return;                                            \
    kernix_par(rows, cols, ^(size_t r0, size_t nr) {                           \
        for (size_t r = r0; r < r0 + nr; r++) {                                \
            const float* x = in + r * cols;                                    \
            float* o = out + r * cols;                                         \
            const int n = (int)cols;                                           \
            (void)n;                                                           \
            __VA_ARGS__;                                                       \
        }                                                                      \
    });                                                                        \
}                                                                              \
static inline float_tensor* name(const float_tensor* a KERNIX_UNPAREN params)  \
{                                                                              \
    if (a == NULL || a->tensor == NULL) return NULL;                           \
    float_tensor* y = kernix_float_new(a->row, a->col, 0);                     \
    if (y == NULL) return NULL;                                                \
    name##_into(a->tensor, y->tensor, (size_t)a->row, (size_t)a->col KERNIX_UNPAREN args); \
    return y;                                                                  \
}

/* torch.softmax(x, dim=-1): 최댓값을 빼서 exp 오버플로 방지 */
KERNIX_ROWS(floats_softmax, KERNIX_GPU_ROWS(0, NULL, NULL, 0.0f), (), (),
    float m, s;
    vDSP_maxv(x, 1, &m, cols); m = -m;
    vDSP_vsadd(x, 1, &m, o, 1, cols); vvexpf(o, o, &n);
    vDSP_sve(o, 1, &s, cols); s = 1.0f / s;
    vDSP_vsmul(o, 1, &s, o, 1, cols))

/* torch.log_softmax(x, dim=-1) = x - max - log(sum(exp(x - max))). _into는 제자리 불가(끝에서 입력을 다시 읽음) */
KERNIX_ROWS(floats_log_softmax, KERNIX_GPU_ROWS(1, NULL, NULL, 0.0f), (), (),
    float m, s;
    vDSP_maxv(x, 1, &m, cols); m = -m;
    vDSP_vsadd(x, 1, &m, o, 1, cols); vvexpf(o, o, &n);
    vDSP_sve(o, 1, &s, cols); s = m - logf(s);
    vDSP_vsadd(x, 1, &s, o, 1, cols))

KERNIX_ROWS(kernix_layer_norm_rows, KERNIX_GPU_ROWS(2, g, bt, eps), (, const float* g, const float* bt, float eps),
    (, g, bt, eps),
    float mean, ss;
    vDSP_meanv(x, 1, &mean, cols); mean = -mean;
    vDSP_vsadd(x, 1, &mean, o, 1, cols);
    vDSP_svesq(o, 1, &ss, cols);
    float rstd = 1.0f / sqrtf(ss / (float)cols + eps); /* torch와 같은 편향 분산(÷N) */
    vDSP_vsmul(o, 1, &rstd, o, 1, cols);
    if (g && bt) vDSP_vma(o, 1, g, 1, bt, 1, o, 1, cols);
    else if (g) vDSP_vmul(o, 1, g, 1, o, 1, cols);
    else if (bt) vDSP_vadd(o, 1, bt, 1, o, 1, cols))

/* torch.nn.functional.layer_norm(x, (cols,), gamma, beta, eps). gamma/beta는 1×cols 또는 NULL */
static inline float_tensor* floats_layer_norm(const float_tensor* a, const float_tensor* gamma,
                                              const float_tensor* beta, float eps)
{
    if ((gamma && (gamma->tensor == NULL || gamma->row != 1 || a == NULL || gamma->col != a->col)) ||
        (beta && (beta->tensor == NULL || beta->row != 1 || a == NULL || beta->col != a->col))) {
        fprintf(stderr, "layer_norm: gamma/beta는 1×cols여야 함\n");
        return NULL;
    }
    const float* g = gamma ? gamma->tensor : NULL;
    const float* bt = beta ? beta->tensor : NULL;
    return kernix_layer_norm_rows(a, g, bt, eps);
}
/* 실제 메모리 재배치(torch의 x.t().contiguous()) */
static inline float_tensor* floats_transpose(const float_tensor* a)
{
    if (a == NULL || a->tensor == NULL) return NULL;
    float_tensor* y = kernix_float_new(a->col, a->row, 0);
    if (y == NULL) return NULL;
    /* 큰 행렬: 16행 블록마다 출력 행을 연속으로 씀 (M1 실측: 1024²에서 vDSP_mtrans 대비 1.4배) */
    const size_t R = (size_t)a->row, C = (size_t)a->col, T = 16;
    const float* in = a->tensor;
    float* out = y->tensor;
    if (R * C <= ((size_t)1 << 18)) { /* M1 실측: 512² 이하는 vDSP 단일 스레드가 가장 빠름 */
        vDSP_mtrans(in, 1, out, 1, (vDSP_Length)C, (vDSP_Length)R);
        return y;
    }
#if defined(KERNIX_USE_METAL)
    if (kernix_metal_transpose(in, out, R, C)) return y;
#endif
    kernix_par((R + T - 1) / T, T * C, ^(size_t b0, size_t nb) {
        size_t i0 = b0 * T, i1 = (b0 + nb) * T < R ? (b0 + nb) * T : R;
        for (size_t j = 0; j < C; j++) {
            float* dst = out + j * R;
            for (size_t i = i0; i < i1; i++) dst[i] = in[i * C + j];
        }
    });
    return y;
}
#endif /* KERNIX_ACCELERATE_ENABLED */

static inline int_tensor* ints_add(const int_tensor* a, const int_tensor* b) {
    if (a == NULL || b == NULL || a->tensor == NULL || b->tensor == NULL) {
        return NULL;
    }
    if (a->row != b->row || a->col != b->col) {
        fprintf(stderr, "덧셈 차원 불일치: A(%d×%d) vs B(%d×%d)\n", a->row, a->col, b->row, b->col);
        return NULL;
    }

    int_tensor* p = int_zeros(a->row, a->col);
    if (p == NULL || p->tensor == NULL) return NULL;
    size_t count = (size_t)a->row * (size_t)a->col;

    for (size_t i = 0; i < count; i++) {
        p->tensor[i] = a->tensor[i] + b->tensor[i];
    }
    return p;
}

static inline float_tensor* floats_add(const float_tensor* a, const float_tensor* b) {
#if defined(KERNIX_ACCELERATE_ENABLED)
    return kernix_binary(a, b, KERNIX_ADD); /* 가속 빌드: 병렬 SIMD + 1×cols 행 브로드캐스트 */
#else
    if (a == NULL || b == NULL || a->tensor == NULL || b->tensor == NULL) {
        return NULL;
    }
    if (a->row != b->row || a->col != b->col) {
        fprintf(stderr, "덧셈 차원 불일치: A(%d×%d) vs B(%d×%d)\n", a->row, a->col, b->row, b->col);
        return NULL;
    }

    float_tensor* p = float_zeros(a->row, a->col);
    if (p == NULL || p->tensor == NULL) return NULL;
    size_t count = (size_t)a->row * (size_t)a->col;

    for (size_t i = 0; i < count; i++) {
        p->tensor[i] = a->tensor[i] + b->tensor[i];
    }
    return p;
#endif
}

static inline int_tensor* int_matmul(const int_tensor* a, const int_tensor* b) {
    if (a == NULL || b == NULL || a->tensor == NULL || b->tensor == NULL) {
        return NULL;
    }
    if (a->col != b->row) {
        fprintf(stderr, "행렬 곱셈 차원 불일치: A(%d×%d) × B(%d×%d)\n", a->row, a->col, b->row, b->col);
        return NULL;
    }

    int_tensor* p = int_zeros(a->row, b->col);
    if (p == NULL || p->tensor == NULL) return NULL;

    for (int i = 0; i < a->row; i++) {
        for (int k = 0; k < a->col; k++) {
            int a_val = a->tensor[i * a->col + k];
            for (int j = 0; j < b->col; j++) {
                p->tensor[i * p->col + j] += a_val * b->tensor[k * b->col + j];
            }
        }
    }
    return p;
}

static inline float_tensor* float_matmul(const float_tensor* a, const float_tensor* b) {
    if (a == NULL || b == NULL || a->tensor == NULL || b->tensor == NULL) {
        return NULL;
    }
    if (a->col != b->row) {
        fprintf(stderr, "행렬 곱셈 차원 불일치: A(%d×%d) × B(%d×%d)\n", a->row, a->col, b->row, b->col);
        return NULL;
    }

    float_tensor* p = float_zeros(a->row, b->col);
    if (p == NULL || p->tensor == NULL) return NULL;

    for (int i = 0; i < a->row; i++) {
        for (int k = 0; k < a->col; k++) {
            float a_val = a->tensor[i * a->col + k];
            for (int j = 0; j < b->col; j++) {
                p->tensor[i * p->col + j] += a_val * b->tensor[k * b->col + j];
            }
        }
    }
    return p;
}

/* 텐서 전체 합 (double로 누적). 실패 시 -1 */
static inline int kernix_thread_pool_sum(const float_tensor* a, double* result)
{
    if (a == NULL || a->tensor == NULL || result == NULL || a->row < 0 || a->col < 0) {
        return -1;
    }
    kernix_sum_ctx x = { a->tensor, (size_t)a->row * (size_t)a->col, NULL };
    size_t chunks = (x.n + KERNIX_SUM_CHUNK - 1) / KERNIX_SUM_CHUNK;
    double stack_part[64];
    x.part = chunks <= 64 ? stack_part : malloc(chunks * sizeof(double));
    if (x.part == NULL) {
        return -1;
    }
    int rc = kernix_pool_run(chunks, kernix_sum_chunk, &x);
    double total = 0.0;
    for (size_t i = 0; rc == 0 && i < chunks; i++) {
        total += x.part[i];
    }
    if (x.part != stack_part) {
        free(x.part);
    }
    if (rc != 0) {
        return -1;
    }
    *result = total;
    return 0;
}

/* ── 순수 스레드 풀 기반 병렬 행렬 곱셈 (float) ── */
static inline float_tensor* float_matmul_native_parallel(const float_tensor* a,
                                                          const float_tensor* b)
{
    if (a == NULL || b == NULL ||
        a->tensor == NULL || b->tensor == NULL) {
        return NULL;
    }
    if (a->col != b->row) {
        fprintf(stderr,
            "행렬 곱셈 차원 불일치: A(%d×%d) × B(%d×%d)\n",
            a->row, a->col, b->row, b->col);
        return NULL;
    }

    float_tensor* c = float_zeros(a->row, b->col);
    if (c == NULL || c->tensor == NULL) {
        return NULL;
    }

    kernix_mm_ctx x = { a->tensor, b->tensor, c->tensor, a->col, b->col, a->row, 0 };
    if (kernix_pool_run(kernix_mm_split(&x), kernix_float_mm_chunk, &x) != 0) {
        float_free(c);
        return NULL;
    }
    return c;
}

/* ── 병렬/가속 행렬 곱셈 (float) ── */
static inline float_tensor* float_matmul_parallel(const float_tensor* a,
                                                   const float_tensor* b)
{
#if defined(KERNIX_ACCELERATE_ENABLED)
    if (a == NULL || b == NULL ||
        a->tensor == NULL || b->tensor == NULL) {
        return NULL;
    }
    if (a->col != b->row) {
        fprintf(stderr,
            "행렬 곱셈 차원 불일치: A(%d×%d) × B(%d×%d)\n",
            a->row, a->col, b->row, b->col);
        return NULL;
    }

    float_tensor* c = float_zeros(a->row, b->col);
    if (c == NULL || c->tensor == NULL) {
        return NULL;
    }

    if (a->row > 0 && a->col > 0 && b->col > 0) { /* cblas는 lda/ldb=0 거부, 빈 결과는 이미 0 */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    a->row, b->col, a->col,
                    1.0f, a->tensor, a->col,
                    b->tensor, b->col,
                    0.0f, c->tensor, b->col);
    }
    return c;
#else
    return float_matmul_native_parallel(a, b);
#endif
}

/* ── 스레드 풀 기반 병렬 행렬 곱셈 (int) ── */
static inline int_tensor* int_matmul_parallel(const int_tensor* a,
                                               const int_tensor* b)
{
    if (a == NULL || b == NULL ||
        a->tensor == NULL || b->tensor == NULL) {
        return NULL;
    }
    if (a->col != b->row) {
        fprintf(stderr,
            "행렬 곱셈 차원 불일치: A(%d×%d) × B(%d×%d)\n",
            a->row, a->col, b->row, b->col);
        return NULL;
    }

    int_tensor* c = int_zeros(a->row, b->col);
    if (c == NULL || c->tensor == NULL) {
        return NULL;
    }

    kernix_mm_ctx x = { a->tensor, b->tensor, c->tensor, a->col, b->col, a->row, 0 };
    if (kernix_pool_run(kernix_mm_split(&x), kernix_int_mm_chunk, &x) != 0) {
        int_free(c);
        return NULL;
    }
    return c;
}


/* 활성화 번호 (.knx 파일 형식과 같음, docs/mlp-runtime-design.md) */
enum {
    KERNIX_ACT_NONE = -1, KERNIX_ACT_RELU, KERNIX_ACT_GELU, KERNIX_ACT_SILU, KERNIX_ACT_SIGMOID, KERNIX_ACT_TANH, /* KNX1/2 */
    KERNIX_ACT_LEAKY_RELU, /* x < 0 ? alpha·x : x */
    KERNIX_ACT_ELU,        /* x < 0 ? alpha·(eˣ − 1) : x */
    KERNIX_ACT_SELU,
    KERNIX_ACT_SOFTPLUS,   /* log(1 + eˣ) */
    KERNIX_ACT_RELU6,
    KERNIX_ACT_HARD_SIGMOID, /* relu6(x + 3) / 6 (Keras 3, PyTorch) */
    KERNIX_ACT_HARD_SWISH,   /* x · relu6(x + 3) / 6 */
    KERNIX_ACT_GELU_TANH,    /* tanh 근사 GELU */
    KERNIX_ACT_EXP,
    KERNIX_ACT_MISH,         /* x · tanh(softplus(x)) */
    KERNIX_ACT_SOFTSIGN,     /* x / (1 + |x|) */
    KERNIX_ACT_LAST = KERNIX_ACT_SOFTSIGN
};

#if defined(KERNIX_ACCELERATE_ENABLED)
/* 행 [r0, r1)에 대해 호출자가 하는 후처리 */
typedef void (^kernix_rows_fn)(size_t r0, size_t r1);

/* c = act(a × b + bias)를 호출자가 할당한 c(a->row × b->col)에 씀. bias: b->col개 또는 NULL, act: KERNIX_ACT_*.
 * GPU가 맡은 행은 GPU에서 bias·act까지 끝내고, CPU가 계산한 행 [r0, r1)은 그 직후 cpu_ep(r0, r1)로 넘긴다
 * (호출자가 CPU에서 bias·act를 적용. NULL이면 CPU 행에는 후처리 없음). 성공 0, 실패 -1 */
#if defined(KERNIX_USE_METAL)
int kernix_matmul_ep_into(const float_tensor* a, const float_tensor* b, float_tensor* c,
                          const float* bias, int act, float alpha, kernix_rows_fn cpu_ep);
#else
static inline int kernix_matmul_ep_into(const float_tensor* a, const float_tensor* b, float_tensor* c,
                                        const float* bias, int act, float alpha, kernix_rows_fn cpu_ep)
{
    (void)bias; (void)act; (void)alpha; /* GPU가 없으면 모든 행을 cpu_ep가 처리 */
    if (a == NULL || b == NULL || c == NULL || a->tensor == NULL || b->tensor == NULL || c->tensor == NULL) {
        return -1;
    }
    if (a->col != b->row) {
        fprintf(stderr, "행렬 곱셈 차원 불일치: A(%d×%d) × B(%d×%d)\n", a->row, a->col, b->row, b->col);
        return -1;
    }
    if (c->row != a->row || c->col != b->col) return -1;
    if (a->row == 0 || b->col == 0) return 0;
    if (a->col == 0) {
        memset(c->tensor, 0, sizeof(float) * (size_t)c->row * (size_t)c->col);
    } else {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a->row, b->col, a->col,
                    1.0f, a->tensor, a->col, b->tensor, b->col, 0.0f, c->tensor, b->col);
    }
    if (cpu_ep != NULL) cpu_ep(0, (size_t)a->row);
    return 0;
}
#endif

/* C = A × B를 호출자가 할당한 c(a->row × b->col)에 씀. 성공 0, 실패 -1 */
static inline int kernix_matmul_into(const float_tensor* a, const float_tensor* b, float_tensor* c)
{
    return kernix_matmul_ep_into(a, b, c, NULL, KERNIX_ACT_NONE, 0.0f, NULL);
}
#endif

#if defined(KERNIX_USE_METAL)
/* Metal GPU 행렬 곱셈 (src/kernix_metal.m, MPS). 실패 시 NULL */
float_tensor* float_matmul_metal(const float_tensor* a, const float_tensor* b);
/* 융합 원소 연산 y = relu(a*x + b): 메모리를 한 번만 읽고 씀. 크면 Metal GPU, 작으면 CPU. 실패 시 NULL */
float_tensor* floats_fma_relu_metal(const float_tensor* a, const float_tensor* x, const float_tensor* b);
#endif

#define show_ints(tensor) show_ints_impl(tensor,#tensor)

#define show_floats(tensor) show_floats_impl(tensor,#tensor)

#define show_in8ts(tensor) show_in8ts_impl(tensor,#tensor,0)

#define show_in8ts_bits(tensor) show_in8ts_impl(tensor,#tensor,1)
#endif

