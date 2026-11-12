#ifndef KERNIX_USE_METAL
#define KERNIX_USE_METAL
#endif
#ifndef KERNIX_USE_ACCELERATE
#define KERNIX_USE_ACCELERATE
#endif
#include "kernix.h"
#include <stdatomic.h>
#include <mach/mach_time.h>
#include <os/lock.h>

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

static id<MTLDevice> kernix_dev;
static id<MTLCommandQueue> kernix_queue;

static int kernix_metal_init(void)
{
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        kernix_dev = MTLCreateSystemDefaultDevice();
        if (kernix_dev && MPSSupportsMTLDevice(kernix_dev)) {
            kernix_queue = [kernix_dev newCommandQueue];
        }
    });
    return kernix_queue != nil;
}

/* float_zeros가 Metal로 할당한 메모리: 시작 주소 -> 버퍼 */
static NSMutableDictionary<NSValue*, id<MTLBuffer>>* kernix_bufs;
/* 해제된 버퍼 재사용 캐시: 새 Metal 버퍼 생성(16MB에 ~1ms)을 피함
 * shortcut: 같은 크기만 재사용, 상한을 넘으면 오래된 것부터 해제. 크기가 다양하면 크기 구간별 풀로 확장 */
static NSMutableArray<id<MTLBuffer>>* kernix_free_bufs;
static size_t kernix_free_bytes;
#define KERNIX_CACHE_MAX ((size_t)256 << 20)
static os_unfair_lock kernix_bufs_lock = OS_UNFAIR_LOCK_INIT;

void* kernix_metal_alloc(size_t bytes)
{
    if (!kernix_metal_init()) return NULL;
    @autoreleasepool {
        id<MTLBuffer> buf = nil;
        os_unfair_lock_lock(&kernix_bufs_lock);
        if (kernix_bufs == nil) {
            kernix_bufs = [NSMutableDictionary dictionary];
            kernix_free_bufs = [NSMutableArray array];
        }
        for (NSUInteger i = 0; i < kernix_free_bufs.count; i++) {
            if (kernix_free_bufs[i].length == bytes) {
                buf = kernix_free_bufs[i];
                [kernix_free_bufs removeObjectAtIndex:i];
                kernix_free_bytes -= bytes;
                break;
            }
        }
        os_unfair_lock_unlock(&kernix_bufs_lock);

        if (buf == nil) buf = [kernix_dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (buf == nil) return NULL;

        os_unfair_lock_lock(&kernix_bufs_lock);
        kernix_bufs[[NSValue valueWithPointer:buf.contents]] = buf;
        os_unfair_lock_unlock(&kernix_bufs_lock);
        return buf.contents;
    }
}

static id<MTLBuffer> kernix_lookup(void* p, int remove)
{
    NSValue* key = [NSValue valueWithPointer:p];
    os_unfair_lock_lock(&kernix_bufs_lock);
    id<MTLBuffer> buf = kernix_bufs[key];
    if (remove && buf != nil) {
        [kernix_bufs removeObjectForKey:key];
        if (buf.length <= KERNIX_CACHE_MAX) {
            while (kernix_free_bytes + buf.length > KERNIX_CACHE_MAX) {
                kernix_free_bytes -= kernix_free_bufs[0].length;
                [kernix_free_bufs removeObjectAtIndex:0];
            }
            [kernix_free_bufs addObject:buf];
            kernix_free_bytes += buf.length;
        } else {
            [buf setPurgeableState:MTLPurgeableStateEmpty]; /* 캐시에 안 넣는 큰 버퍼: 페이지를 바로 버림 */
        }
    }
    os_unfair_lock_unlock(&kernix_bufs_lock);
    return buf;
}

int kernix_metal_free(void* p)
{
    @autoreleasepool {
        return p != NULL && kernix_lookup(p, 1) != nil;
    }
}

void kernix_metal_release_cache(void)
{
    @autoreleasepool {
        os_unfair_lock_lock(&kernix_bufs_lock);
        for (id<MTLBuffer> b in kernix_free_bufs) [b setPurgeableState:MTLPurgeableStateEmpty]; /* 해제를 미루지 않고 페이지를 바로 버림 */
        [kernix_free_bufs removeAllObjects];
        kernix_free_bytes = 0;
        os_unfair_lock_unlock(&kernix_bufs_lock);
    }
}

/* 페이지 정렬 메모리는 복사 없이 GPU 버퍼로 감싸고, 아니면 복사본 버퍼를 만든다 */
static id<MTLBuffer> kernix_wrap(float* p, size_t bytes, int* copied)
{
    *copied = ((uintptr_t)p % KERNIX_PAGE) != 0;
    if (!*copied) {
        size_t len = (bytes + KERNIX_PAGE - 1) / KERNIX_PAGE * KERNIX_PAGE;
        return [kernix_dev newBufferWithBytesNoCopy:p length:len
                                            options:MTLResourceStorageModeShared
                                        deallocator:nil];
    }
    return [kernix_dev newBufferWithBytes:p length:bytes options:MTLResourceStorageModeShared];
}

/* Metal 할당은 그대로, 외부 페이지 정렬 메모리는 호출마다 no-copy로 감싸고, 나머지는 복사 */
static id<MTLBuffer> kernix_buffer(float* p, size_t bytes, int* copied)
{
    *copied = 0;
    id<MTLBuffer> buf = kernix_lookup(p, 0);
    return buf != nil ? buf : kernix_wrap(p, bytes, copied);
}

/* GPU가 맡는 행 비율, log2(M·N·K) 구간별로 실측 보정 (칩마다 GPU 지연/성능비가 다름).
 * 0 = 아직 측정 안 함(0.6에서 시작). 최소값까지 내려간 구간은 GPU 지연이 이득보다 커서 GPU를 끈다.
 * shortcut: 꺼진 구간은 프로세스가 끝날 때까지 다시 시도하지 않음, 부하가 크게 바뀌는 환경이면 주기적 재시도 추가 */
#define KERNIX_FRAC_MIN 0.05
static _Atomic double kernix_gpu_frac[64];

/* GPUEndTime과 같은 기준(mach_absolute_time)의 초 단위 시각 */
static double kernix_host_sec(void)
{
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom * 1e-9;
}

static void kernix_cpu_rows(const float_tensor* a, const float_tensor* b, float_tensor* c,
                            int r0, int r1)
{
    if (r1 <= r0) return;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                r1 - r0, b->col, a->col,
                1.0f, a->tensor + (size_t)r0 * a->col, a->col,
                b->tensor, b->col,
                0.0f, c->tensor + (size_t)r0 * c->col, c->col);
}

/* C의 앞쪽 행은 GPU(MPS), 나머지 행은 CPU AMX(Accelerate)가 동시에 계산.
 * 통합 메모리라 같은 C 버퍼를 복사 없이 나눠 쓴다. GPU가 없거나 실패하면 CPU로 마저 계산.
 * GPU 행의 bias·act는 같은 명령 버퍼에 커널로 이어 붙이고, CPU 행은 계산 직후 cpu_ep에 넘긴다 (kernix.h).
 * c는 호출자가 할당한 a->row × b->col 텐서. 성공 0, 실패 -1 */
static id<MTLComputePipelineState> kernix_ep_pso(void);

int kernix_matmul_ep_into(const float_tensor* a, const float_tensor* b, float_tensor* c,
                          const float* bias, int act, float alpha, kernix_rows_fn cpu_ep)
{
    if (a == NULL || b == NULL || c == NULL || a->tensor == NULL || b->tensor == NULL || c->tensor == NULL) {
        return -1;
    }
    if (a->col != b->row) {
        fprintf(stderr, "행렬 곱셈 차원 불일치: A(%d×%d) × B(%d×%d)\n",
                a->row, a->col, b->row, b->col);
        return -1;
    }
    if (c->row != a->row || c->col != b->col) return -1;
    if (a->row == 0 || b->col == 0) return 0;
    if (a->col == 0) { /* 내부 차원 0: 곱은 전부 0 (c는 초기화 안 된 메모리일 수 있음), 후처리는 CPU에서 */
        memset(c->tensor, 0, sizeof(float) * (size_t)c->row * (size_t)c->col);
        if (cpu_ep != nil) cpu_ep(0, (size_t)a->row);
        return 0;
    }
    const int ep = bias != NULL || act != KERNIX_ACT_NONE;

    NSUInteger m = (NSUInteger)a->row, k = (NSUInteger)a->col, n = (NSUInteger)b->col;

    /* 작은 행렬은 GPU 실행 지연(~0.2ms)이 계산보다 커서 AMX만 사용 */
    unsigned long long mnk = (unsigned long long)m * n * k;
    _Atomic double* frac = &kernix_gpu_frac[63 - __builtin_clzll(mnk)];
    double f0 = atomic_load(frac);
    if (f0 == 0.0) f0 = 0.6;
    NSUInteger mg = 0;
    /* shortcut: M1 실측으로 정한 하한(512³에선 GPU 지연 때문에 AMX 단독이 더 빠름), 다른 칩이면 재측정 */
    if (mnk >= (1ull << 28) && f0 > KERNIX_FRAC_MIN && kernix_metal_init() &&
        (!ep || (kernix_ep_pso() != nil && (unsigned long long)m * n < UINT32_MAX))) {
        mg = (NSUInteger)((double)m * f0);
    }

    int gpu_ok = 0;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = nil;
        double t0 = kernix_host_sec();

        if (mg > 0) {
            int ca, cb, cc;
            id<MTLBuffer> ba = kernix_buffer(a->tensor, m * k * sizeof(float), &ca);
            id<MTLBuffer> bb = kernix_buffer(b->tensor, k * n * sizeof(float), &cb);
            id<MTLBuffer> bc = kernix_buffer(c->tensor, m * n * sizeof(float), &cc); /* float_zeros라 항상 no-copy */

            if (ba && bb && bc && !cc) {
                MPSMatrix* ma = [[MPSMatrix alloc] initWithBuffer:ba descriptor:
                    [MPSMatrixDescriptor matrixDescriptorWithRows:mg columns:k
                                                         rowBytes:k * sizeof(float) dataType:MPSDataTypeFloat32]];
                MPSMatrix* mb = [[MPSMatrix alloc] initWithBuffer:bb descriptor:
                    [MPSMatrixDescriptor matrixDescriptorWithRows:k columns:n
                                                         rowBytes:n * sizeof(float) dataType:MPSDataTypeFloat32]];
                MPSMatrix* mc = [[MPSMatrix alloc] initWithBuffer:bc descriptor:
                    [MPSMatrixDescriptor matrixDescriptorWithRows:mg columns:n
                                                         rowBytes:n * sizeof(float) dataType:MPSDataTypeFloat32]];
                MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
                    initWithDevice:kernix_dev transposeLeft:NO transposeRight:NO
                        resultRows:mg resultColumns:n interiorColumns:k alpha:1.0 beta:0.0];

                cmd = [kernix_queue commandBuffer];
                [mm encodeToCommandBuffer:cmd leftMatrix:ma rightMatrix:mb resultMatrix:mc];
                if (ep) { /* 같은 명령 버퍼라 행렬곱이 끝나면 바로 실행 (CPU 왕복 없음) */
                    int cbias;
                    id<MTLBuffer> bbias = bias != NULL ? kernix_buffer((float*)bias, n * sizeof(float), &cbias) : bc;
                    uint32_t cfg[4] = { (uint32_t)act, (uint32_t)n, bias != NULL, 0 };
                    memcpy(&cfg[3], &alpha, sizeof alpha);
                    id<MTLComputePipelineState> pso = kernix_ep_pso();
                    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                    [enc setComputePipelineState:pso];
                    [enc setBuffer:bc offset:0 atIndex:0];
                    [enc setBuffer:bbias offset:0 atIndex:1];
                    [enc setBytes:cfg length:sizeof cfg atIndex:2];
                    [enc dispatchThreads:MTLSizeMake(mg * n, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(pso.maxTotalThreadsPerThreadgroup, 1, 1)];
                    [enc endEncoding];
                }
                [cmd commit];
            }
        }

        /* GPU가 도는 동안 CPU는 나머지 행 계산 */
        kernix_cpu_rows(a, b, c, (int)mg, (int)m);
        if (cpu_ep != nil && mg < m) cpu_ep(mg, m);
        double t_cpu = kernix_host_sec() - t0; /* 후처리까지 포함해 비율 보정 (GPU 쪽도 후처리 포함) */

        if (cmd != nil) {
            [cmd waitUntilCompleted];
            gpu_ok = cmd.status == MTLCommandBufferStatusCompleted;
        }

        if (gpu_ok && mg < m) {
            double r_gpu = (double)mg / (cmd.GPUEndTime - t0);
            double r_cpu = (double)(m - mg) / t_cpu;
            double f = 0.5 * f0 + 0.5 * r_gpu / (r_gpu + r_cpu);
            atomic_store(frac, f < KERNIX_FRAC_MIN ? KERNIX_FRAC_MIN : f > 0.95 ? 0.95 : f);
        }
    }

    if (mg > 0 && !gpu_ok) {
        fprintf(stderr, "Metal GPU 실패, CPU로 계산\n");
        kernix_cpu_rows(a, b, c, 0, (int)mg);
        if (cpu_ep != nil) cpu_ep(0, mg);
    }
    return 0;
}

/* 행렬곱 결과 위에서 제자리로 bias + 활성화. cfg: x = KERNIX_ACT_* (-1 = 없음), y = 열 수, z = bias 있음, w = alpha(float 비트).
 * 수식은 kernix_model.h kernix_act_chunk(CPU)와 같다 */
static NSString* const kernix_ep_src =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "static float relu6(float v) { return min(max(v, 0.0f), 6.0f); }\n"
     "static float softplus(float v) { return v > 20.0f ? v : log(1.0f + exp(v)); }\n"
     "kernel void bias_act(device float* y [[buffer(0)]], device const float* b [[buffer(1)]],\n"
     "                     constant uint4& cfg [[buffer(2)]], uint i [[thread_position_in_grid]]) {\n"
     "    float v = y[i], a = as_type<float>(cfg.w);\n"
     "    if (cfg.z) v += b[i % cfg.y];\n"
     "    switch ((int)cfg.x) {\n"
     "    case 0: v = max(v, 0.0f); break;\n"
     "    case 1: { float z = fabs(v) * 0.70710678f, k = 1.0f / (1.0f + 0.3275911f * z);\n"
     "              float p = k * (0.254829592f + k * (-0.284496736f + k * (1.421413741f + k * (-1.453152027f + k * 1.061405429f))));\n"
     "              v = 0.5f * v * (1.0f + copysign(1.0f - p * exp(-z * z), v)); break; }\n"
     "    case 2: v = v / (1.0f + exp(-v)); break;\n"
     "    case 3: v = 1.0f / (1.0f + exp(-v)); break;\n"
     "    case 4: v = tanh(v); break;\n"
     "    case 5: v = v < 0.0f ? a * v : v; break;\n"
     "    case 6: v = v < 0.0f ? a * (exp(v) - 1.0f) : v; break;\n"
     "    case 7: v = 1.0507009873554805f * (v < 0.0f ? 1.6732632423543772f * (exp(v) - 1.0f) : v); break;\n"
     "    case 8: v = softplus(v); break;\n"
     "    case 9: v = relu6(v); break;\n"
     "    case 10: v = relu6(v + 3.0f) / 6.0f; break;\n"
     "    case 11: v = v * relu6(v + 3.0f) / 6.0f; break;\n"
     "    case 12: v = 0.5f * v * (1.0f + tanh(0.7978845608028654f * (v + 0.044715f * v * v * v))); break;\n"
     "    case 13: v = exp(v); break;\n"
     "    case 14: v = v * tanh(softplus(v)); break;\n"
     "    case 15: v = v / (1.0f + fabs(v)); break;\n"
     "    }\n"
     "    y[i] = v;\n"
     "}\n";

static id<MTLComputePipelineState> kernix_ep_pso(void)
{
    static id<MTLComputePipelineState> pso;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.mathMode = MTLMathModeSafe; /* fast-math 끔: CPU 경로와 같은 정밀도 */
        opts.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        NSError* err = nil;
        id<MTLLibrary> lib = [kernix_dev newLibraryWithSource:kernix_ep_src options:opts error:&err];
        if (lib != nil) {
            pso = [kernix_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bias_act"] error:&err];
        }
        if (pso == nil) fprintf(stderr, "Metal 커널 컴파일 실패: %s\n", err.description.UTF8String);
    });
    return pso;
}

float_tensor* float_matmul_metal(const float_tensor* a, const float_tensor* b)
{
    if (a == NULL || b == NULL) return NULL;
    float_tensor* c = kernix_float_new(a->row, b->col, 0); /* beta=0이라 모든 원소를 덮어씀 */
    if (c != NULL && kernix_matmul_into(a, b, c) != 0) {
        float_free(c);
        c = NULL;
    }
    return c;
}

/* ── 융합 원소 연산 ── */
static NSString* const kernix_ew_src =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void fma_relu(device const float4* a [[buffer(0)]], device const float4* x [[buffer(1)]],\n"
     "                     device const float4* b [[buffer(2)]], device float4* y [[buffer(3)]],\n"
     "                     uint i [[thread_position_in_grid]]) {\n"
     "    y[i] = max(fma(a[i], x[i], b[i]), 0.0f);\n"
     "}\n";

static id<MTLComputePipelineState> kernix_fma_relu_pso(void)
{
    static id<MTLComputePipelineState> pso;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        NSError* err = nil;
        id<MTLLibrary> lib = [kernix_dev newLibraryWithSource:kernix_ew_src options:nil error:&err];
        if (lib != nil) {
            pso = [kernix_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fma_relu"] error:&err];
        }
        if (pso == nil) fprintf(stderr, "Metal 커널 컴파일 실패: %s\n", err.description.UTF8String);
    });
    return pso;
}

static void kernix_fma_relu_cpu(const float* a, const float* x, const float* b, float* y, size_t s, size_t e)
{
    for (size_t i = s; i < e; i++) {
        float v = a[i] * x[i] + b[i];
        y[i] = v > 0.0f ? v : 0.0f;
    }
}

float_tensor* floats_fma_relu_metal(const float_tensor* a, const float_tensor* x, const float_tensor* b)
{
    if (a == NULL || x == NULL || b == NULL ||
        a->tensor == NULL || x->tensor == NULL || b->tensor == NULL) {
        return NULL;
    }
    if (a->row != x->row || a->col != x->col || a->row != b->row || a->col != b->col) {
        fprintf(stderr, "원소 연산 차원 불일치: %d×%d, %d×%d, %d×%d\n",
                a->row, a->col, x->row, x->col, b->row, b->col);
        return NULL;
    }

    float_tensor* y = kernix_float_new(a->row, a->col, 0); /* 모든 원소를 덮어씀 */
    if (y == NULL) return NULL;
    size_t count = (size_t)a->row * (size_t)a->col;
    size_t n4 = count / 4;

    /* shortcut: M1 실측 하한(4M 원소까지는 CPU 단일 스레드가 GPU 지연 포함 시간보다 빠름), 다른 칩이면 재측정 */
    int gpu = count >= ((size_t)1 << 23) && kernix_metal_init() && kernix_fma_relu_pso() != nil;
    int ok = 0;

    if (gpu) {
        @autoreleasepool {
            int c0, c1, c2, c3;
            size_t bytes = count * sizeof(float);
            id<MTLBuffer> ba = kernix_buffer(a->tensor, bytes, &c0);
            id<MTLBuffer> bx = kernix_buffer(x->tensor, bytes, &c1);
            id<MTLBuffer> bb = kernix_buffer(b->tensor, bytes, &c2);
            id<MTLBuffer> by = kernix_buffer(y->tensor, bytes, &c3); /* 자체 할당이라 항상 no-copy */

            if (ba && bx && bb && by && !c3) {
                id<MTLComputePipelineState> pso = kernix_fma_relu_pso();
                id<MTLCommandBuffer> cmd = [kernix_queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:ba offset:0 atIndex:0];
                [enc setBuffer:bx offset:0 atIndex:1];
                [enc setBuffer:bb offset:0 atIndex:2];
                [enc setBuffer:by offset:0 atIndex:3];
                [enc dispatchThreads:MTLSizeMake(n4, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(pso.maxTotalThreadsPerThreadgroup, 1, 1)];
                [enc endEncoding];
                [cmd commit];
                /* GPU가 도는 동안 float4로 나누어떨어지지 않는 꼬리 원소는 CPU가 처리 */
                kernix_fma_relu_cpu(a->tensor, x->tensor, b->tensor, y->tensor, n4 * 4, count);
                [cmd waitUntilCompleted];
                ok = cmd.status == MTLCommandBufferStatusCompleted;
            }
        }
        if (!ok) fprintf(stderr, "Metal GPU 실패, CPU로 계산\n");
    }
    if (!ok) {
        if (count < ((size_t)1 << 18)) {
            kernix_fma_relu_cpu(a->tensor, x->tensor, b->tensor, y->tensor, 0, count);
        } else {
            /* 중간 크기: 코어 수만큼 나눠 메모리 대역폭을 채움 (GCD) */
            const size_t chunks = 8;
            dispatch_apply(chunks, DISPATCH_APPLY_AUTO, ^(size_t c) {
                kernix_fma_relu_cpu(a->tensor, x->tensor, b->tensor, y->tensor,
                                    c * count / chunks, (c + 1) * count / chunks);
            });
        }
    }
    return y;
}

/* ── 원소별 연산 (kernix.h KERNIX_GPU_* 번호와 일치) ── */
static NSString* const kernix_map_src =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "static float4 gelu4(float4 x) {\n"   /* CPU 경로와 같은 Abramowitz-Stegun erf */
     "    float4 z = fabs(x) * 0.70710678f, k = 1.0f / (1.0f + 0.3275911f * z);\n"
     "    float4 p = k * (0.254829592f + k * (-0.284496736f + k * (1.421413741f + k * (-1.453152027f + k * 1.061405429f))));\n"
     "    return 0.5f * x * (1.0f + copysign(1.0f - p * exp(-z * z), x));\n"
     "}\n"
     "kernel void map_op(device const float4* x [[buffer(0)]], device const float4* b [[buffer(1)]],\n"
     "                   device float4* y [[buffer(2)]], constant uint4& cfg [[buffer(3)]],\n"
     "                   constant float2& p [[buffer(4)]], uint i [[thread_position_in_grid]]) {\n"
     "    float4 v = x[i], w = cfg.y ? b[i % cfg.y] : b[i];\n"   /* cfg: x=op, y=브로드캐스트 행 길이/4 */
     "    switch (cfg.x) {\n"
     "    case 0: y[i] = -v; break;                case 1: y[i] = fabs(v); break;\n"
     "    case 2: y[i] = exp(v); break;            case 3: y[i] = log(v); break;\n"
     "    case 4: y[i] = sqrt(v); break;           case 5: y[i] = tanh(v); break;\n"
     "    case 6: y[i] = max(v, 0.0f); break;      case 7: y[i] = 1.0f / (1.0f + exp(-v)); break;\n"
     "    case 8: y[i] = v / (1.0f + exp(-v)); break; case 9: y[i] = gelu4(v); break;\n"
     "    case 10: y[i] = v + p.x; break;          case 11: y[i] = v * p.x; break;\n"
     "    case 12: y[i] = clamp(v, p.x, p.y); break;\n"
     "    case 13: y[i] = v + w; break;            case 14: y[i] = v - w; break;\n"
     "    case 15: y[i] = v * w; break;            case 16: y[i] = v / w; break;\n"
     "    }\n"
     "}\n";

static id<MTLComputePipelineState> kernix_map_pso(void)
{
    static id<MTLComputePipelineState> pso;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.mathMode = MTLMathModeSafe; /* fast-math 끔: CPU/PyTorch와 같은 정밀도 */
        opts.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        NSError* err = nil;
        id<MTLLibrary> lib = [kernix_dev newLibraryWithSource:kernix_map_src options:opts error:&err];
        if (lib != nil) {
            pso = [kernix_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"map_op"] error:&err];
        }
        if (pso == nil) fprintf(stderr, "Metal 커널 컴파일 실패: %s\n", err.description.UTF8String);
    });
    return pso;
}

/* shortcut: M1 실측 하한(이보다 작으면 CPU 병렬이 GPU 지연 포함 시간보다 빠름), 다른 칩이면 재측정 */
#define KERNIX_MAP_GPU_MIN ((size_t)1 << 22)

size_t kernix_metal_map(int op, const float* x, const float* b, size_t bcols, float* y, size_t n, float p0, float p1)
{
    size_t n4 = n / 4;
    if (n < KERNIX_MAP_GPU_MIN || bcols % 4 != 0 || !kernix_metal_init() || kernix_map_pso() == nil) return 0;

    int ok = 0;
    @autoreleasepool {
        int cx, cb, cy;
        size_t bytes = n * sizeof(float);
        id<MTLBuffer> bx = kernix_buffer((float*)x, bytes, &cx);
        id<MTLBuffer> bb = b ? kernix_buffer((float*)b, (bcols ? bcols : n) * sizeof(float), &cb) : bx;
        id<MTLBuffer> by = kernix_buffer(y, bytes, &cy); /* 자체 할당이라 항상 no-copy */
        if (bx && bb && by && !cy) {
            uint32_t cfg[4] = { (uint32_t)op, (uint32_t)(bcols / 4), 0, 0 };
            float p[2] = { p0, p1 };
            id<MTLComputePipelineState> pso = kernix_map_pso();
            id<MTLCommandBuffer> cmd = [kernix_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:bx offset:0 atIndex:0];
            [enc setBuffer:bb offset:0 atIndex:1];
            [enc setBuffer:by offset:0 atIndex:2];
            [enc setBytes:cfg length:sizeof cfg atIndex:3];
            [enc setBytes:p length:sizeof p atIndex:4];
            [enc dispatchThreads:MTLSizeMake(n4, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(pso.maxTotalThreadsPerThreadgroup, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            ok = cmd.status == MTLCommandBufferStatusCompleted;
        }
    }
    if (!ok) fprintf(stderr, "Metal GPU 실패, CPU로 계산\n");
    return ok ? n4 * 4 : 0;
}

/* ── 행 단위 연산과 전치 ── */
static NSString* const kernix_rows_src =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     /* threadgroup 전체 최댓값/합: simdgroup 리덕션 후 simdgroup 대표값끼리 한 번 더 */
     "static float tg_reduce(float v, bool is_max, threadgroup float* sh, uint sg, uint lane, uint nsg) {\n"
     "    v = is_max ? simd_max(v) : simd_sum(v);\n"
     "    if (lane == 0) sh[sg] = v;\n"
     "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     "    v = lane < nsg ? sh[lane] : (is_max ? -INFINITY : 0.0f);\n"
     "    v = is_max ? simd_max(v) : simd_sum(v);\n"
     "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     "    return v;\n"
     "}\n"
     "kernel void rows_op(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],\n"
     "                    device const float* g [[buffer(2)]], device const float* bt [[buffer(3)]],\n"
     "                    constant uint4& cfg [[buffer(4)]], constant float& eps [[buffer(5)]],\n"
     "                    uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],\n"
     "                    uint nt [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],\n"
     "                    uint lane [[thread_index_in_simdgroup]], uint nsg [[simdgroups_per_threadgroup]]) {\n"
     "    threadgroup float sh[32];\n"
     "    const uint C = cfg.x, mode = cfg.y;\n"   /* cfg: x=cols, y=mode, z=gamma 있음, w=beta 있음 */
     "    device const float* xr = x + (ulong)row * C;\n"
     "    device float* yr = y + (ulong)row * C;\n"
     "    if (mode < 2) {\n"
     "        float m = -INFINITY;\n"
     "        for (uint c = t; c < C; c += nt) m = max(m, xr[c]);\n"
     "        m = tg_reduce(m, true, sh, sg, lane, nsg);\n"
     "        float s = 0.0f;\n"
     "        for (uint c = t; c < C; c += nt) s += exp(xr[c] - m);\n"
     "        s = tg_reduce(s, false, sh, sg, lane, nsg);\n"
     "        if (mode == 0) { float inv = 1.0f / s; for (uint c = t; c < C; c += nt) yr[c] = exp(xr[c] - m) * inv; }\n"
     "        else { float k = m + log(s); for (uint c = t; c < C; c += nt) yr[c] = xr[c] - k; }\n"
     "    } else {\n"
     "        float s = 0.0f;\n"
     "        for (uint c = t; c < C; c += nt) s += xr[c];\n"
     "        float mean = tg_reduce(s, false, sh, sg, lane, nsg) / C;\n"
     "        float v = 0.0f;\n"
     "        for (uint c = t; c < C; c += nt) { float d = xr[c] - mean; v += d * d; }\n"
     "        float rstd = rsqrt(tg_reduce(v, false, sh, sg, lane, nsg) / C + eps);\n"
     "        for (uint c = t; c < C; c += nt) {\n"
     "            float o = (xr[c] - mean) * rstd;\n"
     "            if (cfg.z) o *= g[c];\n"
     "            if (cfg.w) o += bt[c];\n"
     "            yr[c] = o;\n"
     "        }\n"
     "    }\n"
     "}\n"
     /* 32×32 타일(+1 패딩으로 뱅크 충돌 방지): 읽기·쓰기 모두 연속 접근 */
     "kernel void transpose(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],\n"
     "                      constant uint2& rc [[buffer(2)]], uint2 tid [[thread_position_in_threadgroup]],\n"
     "                      uint2 tg [[threadgroup_position_in_grid]]) {\n"
     "    threadgroup float tile[32][33];\n"
     "    const uint R = rc.x, C = rc.y, c = tg.x * 32 + tid.x, r = tg.y * 32 + tid.y;\n"
     "    for (uint k = 0; k < 32; k += 8) if (r + k < R && c < C) tile[tid.y + k][tid.x] = x[(ulong)(r + k) * C + c];\n"
     "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     "    const uint oc = tg.y * 32 + tid.x, orow = tg.x * 32 + tid.y;\n"
     "    for (uint k = 0; k < 32; k += 8) if (orow + k < C && oc < R) y[(ulong)(orow + k) * R + oc] = tile[tid.x][tid.y + k];\n"
     "}\n";

static id<MTLComputePipelineState> kernix_rows_pso(int transpose)
{
    static id<MTLComputePipelineState> pso[2];
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.mathMode = MTLMathModeSafe;
        opts.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        NSError* err = nil;
        id<MTLLibrary> lib = [kernix_dev newLibraryWithSource:kernix_rows_src options:opts error:&err];
        if (lib != nil) {
            pso[0] = [kernix_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rows_op"] error:&err];
            pso[1] = [kernix_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"transpose"] error:&err];
        }
        if (pso[0] == nil || pso[1] == nil) fprintf(stderr, "Metal 커널 컴파일 실패: %s\n", err.description.UTF8String);
    });
    return pso[transpose];
}

/* 인코딩 블록을 받아 명령 버퍼를 실행. 성공 시 1 */
static int kernix_gpu_run(void (^encode)(id<MTLComputeCommandEncoder>))
{
    id<MTLCommandBuffer> cmd = [kernix_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    encode(enc);
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "Metal GPU 실패, CPU로 계산\n");
        return 0;
    }
    return 1;
}

int kernix_metal_rows(int mode, const float* x, const float* g, const float* b, float eps,
                      float* y, size_t rows, size_t cols)
{
    if (rows * cols < KERNIX_MAP_GPU_MIN || cols == 0 || !kernix_metal_init() || kernix_rows_pso(0) == nil) return 0;
    @autoreleasepool {
        int cx, cg, cbt, cy;
        size_t bytes = rows * cols * sizeof(float);
        id<MTLBuffer> bx = kernix_buffer((float*)x, bytes, &cx);
        id<MTLBuffer> by = kernix_buffer(y, bytes, &cy);
        id<MTLBuffer> bg = g ? kernix_buffer((float*)g, cols * sizeof(float), &cg) : bx;
        id<MTLBuffer> bb = b ? kernix_buffer((float*)b, cols * sizeof(float), &cbt) : bx;
        if (!bx || !by || !bg || !bb || cy) return 0;
        id<MTLComputePipelineState> pso = kernix_rows_pso(0);
        uint32_t cfg_v[4] = { (uint32_t)cols, (uint32_t)mode, g != NULL, b != NULL };
        const uint32_t* cfg = cfg_v; /* 블록은 배열을 캡처할 수 없어 포인터로 (동기 실행이라 수명 안전) */
        /* 스레드당 원소 4개 이상, 32의 배수, 최대 1024 */
        NSUInteger nt = ((cols + 3) / 4 + 31) / 32 * 32;
        if (nt > pso.maxTotalThreadsPerThreadgroup) nt = pso.maxTotalThreadsPerThreadgroup;
        return kernix_gpu_run(^(id<MTLComputeCommandEncoder> enc) {
            [enc setComputePipelineState:pso];
            [enc setBuffer:bx offset:0 atIndex:0];
            [enc setBuffer:by offset:0 atIndex:1];
            [enc setBuffer:bg offset:0 atIndex:2];
            [enc setBuffer:bb offset:0 atIndex:3];
            [enc setBytes:cfg length:sizeof cfg_v atIndex:4];
            [enc setBytes:&eps length:sizeof eps atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(nt, 1, 1)];
        });
    }
}

int kernix_metal_transpose(const float* x, float* y, size_t rows, size_t cols)
{
    if (rows * cols < KERNIX_MAP_GPU_MIN || !kernix_metal_init() || kernix_rows_pso(1) == nil) return 0;
    @autoreleasepool {
        int cx, cy;
        size_t bytes = rows * cols * sizeof(float);
        id<MTLBuffer> bx = kernix_buffer((float*)x, bytes, &cx);
        id<MTLBuffer> by = kernix_buffer(y, bytes, &cy);
        if (!bx || !by || cy) return 0;
        id<MTLComputePipelineState> pso = kernix_rows_pso(1);
        uint32_t rc_v[2] = { (uint32_t)rows, (uint32_t)cols };
        const uint32_t* rc = rc_v;
        return kernix_gpu_run(^(id<MTLComputeCommandEncoder> enc) {
            [enc setComputePipelineState:pso];
            [enc setBuffer:bx offset:0 atIndex:0];
            [enc setBuffer:by offset:0 atIndex:1];
            [enc setBytes:rc length:sizeof rc_v atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake((cols + 31) / 32, (rows + 31) / 32, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        });
    }
}
