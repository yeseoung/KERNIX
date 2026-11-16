/* ANE 보조: .knx 그래프의 구간을 Core ML NeuralNetwork 모델로 만들어 뉴럴엔진에서 실행 (-DKERNIX_USE_COREML).
 * 설계: docs/superpowers/specs/2026-11-16-ane-assist-design.md. 구간 찾기와 경로 선택은 include/kernix_model.h */
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include <os/lock.h>
#include "kernix_model.h"

int kernix_coreml_available(void)
{
    if (@available(macOS 13.0, *)) return 1; /* MLPredictionOptions.outputBackings */
    return 0;
}

static int kernix_coreml_act_ok(int act)
{
    switch (act) {
    case KERNIX_ACT_NONE: case KERNIX_ACT_RELU: case KERNIX_ACT_SIGMOID: case KERNIX_ACT_TANH: case KERNIX_ACT_LEAKY_RELU:
    case KERNIX_ACT_ELU: case KERNIX_ACT_SOFTPLUS: case KERNIX_ACT_SOFTSIGN: case KERNIX_ACT_RELU6: case KERNIX_ACT_GELU:
    case KERNIX_ACT_GELU_TANH: case KERNIX_ACT_SILU: case KERNIX_ACT_HARD_SIGMOID: case KERNIX_ACT_HARD_SWISH:
        return 1;
    default:
        return 0;
    }
}

int kernix_coreml_step_ok(const kernix_step* s, const kernix_val* in, const kernix_val* out)
{
    (void)in; (void)out;
    switch (s->kind) {
    case KERNIX_OP_ACT: case KERNIX_OP_CONV: case KERNIX_OP_LINEAR:
        return kernix_coreml_act_ok(s->act);
    case KERNIX_OP_BINARY: /* 브로드캐스트는 KERNIX와 같음: 크기 1인 축(H·W 또는 C)이 늘어남 */
        return s->bop == KERNIX_BIN_ADD || s->bop == KERNIX_BIN_SUB || s->bop == KERNIX_BIN_MUL ||
               s->bop == KERNIX_BIN_MAX || s->bop == KERNIX_BIN_MIN;
    default:
        return 1;
    }
}

/* ---- protobuf 쓰기 (Core ML Model.proto / NeuralNetwork.proto, coremltools 9.0의 *_pb2.py에서 읽은 필드 번호) ----
 * Model: specificationVersion 1, description 2, neuralNetwork 500
 * ModelDescription: input 1, output 10 · FeatureDescription: name 1, type 3 · FeatureType: multiArrayType 5
 * ArrayFeatureType: shape 1 (int64), dataType 2 (FLOAT32 = 65568)
 * NeuralNetwork: layers 1, arrayInputShapeMapping 5 (EXACT_ARRAY_MAPPING = 1)
 * NeuralNetworkLayer: name 1, input 2, output 3, convolution 100, pooling 120, activation 130, padding 200, upsample 210,
 *   scale 245, concat 320, clip 660, gelu 795, minBroadcastable 870, maxBroadcastable 875, addBroadcastable 880,
 *   multiplyBroadcastable 900, subtractBroadcastable 905, transpose 985
 * Convolution: outputChannels 1, kernelChannels 2, nGroups 10, kernelSize 20, stride 30, dilationFactor 40, valid 50,
 *   hasBias 70, weights 90, bias 91 · ValidPadding: paddingAmounts 1 · BorderAmounts: borderAmounts 10
 * EdgeSizes: startEdgeSize 1, endEdgeSize 2 · WeightParams: floatValue 1
 * Pooling: type 1 (MAX 0, AVERAGE 1), kernelSize 10, stride 20, valid 30, avgPoolExcludePadding 50, globalPooling 60
 * Activation: ReLU 10, leakyReLU 15 (alpha 1), tanh 30, sigmoid 40, sigmoidHard 41 (alpha 1, beta 2), ELU 50 (alpha 1),
 *   softsign 60, softplus 70 · Clip: minVal 1, maxVal 2 · Gelu: mode 1 (EXACT 0, TANH_APPROXIMATION 1)
 * Scale: shapeScale 1, scale 2, hasBias 3, shapeBias 4, bias 5 · Padding: constant 1 (value 1), paddingAmounts 10
 * Upsample: scalingFactor 1, mode 5 (NN 0) · Transpose: axes 1 */
typedef struct { uint8_t* p; size_t n, cap; int err; } kernix_pb;

static void pb_put(kernix_pb* b, const void* src, size_t n)
{
    if (b->err) return;
    if (b->n + n > b->cap) {
        size_t cap = b->cap ? b->cap : 256;
        while (cap < b->n + n) cap *= 2;
        uint8_t* p = realloc(b->p, cap);
        if (p == NULL) { b->err = 1; return; }
        b->p = p;
        b->cap = cap;
    }
    memcpy(b->p + b->n, src, n);
    b->n += n;
}

static void pb_varint(kernix_pb* b, uint64_t v)
{
    uint8_t t[10];
    int n = 0;
    do { t[n++] = (uint8_t)((v & 0x7f) | (v > 0x7f ? 0x80 : 0)); v >>= 7; } while (v);
    pb_put(b, t, (size_t)n);
}

static void pb_key(kernix_pb* b, int field, int wire) { pb_varint(b, ((uint64_t)field << 3) | (uint64_t)wire); }
static void pb_uint(kernix_pb* b, int field, uint64_t v) { pb_key(b, field, 0); pb_varint(b, v); }
static void pb_float(kernix_pb* b, int field, float v) { pb_key(b, field, 5); pb_put(b, &v, 4); } /* Apple Silicon은 리틀 엔디언 */

static void pb_bytes(kernix_pb* b, int field, const void* p, size_t n)
{
    pb_key(b, field, 2);
    pb_varint(b, n);
    pb_put(b, p, n);
}

static void pb_str(kernix_pb* b, int field, const char* s) { pb_bytes(b, field, s, strlen(s)); }

/* 하위 메시지 sub를 field로 붙이고 sub를 비움 */
static void pb_msg(kernix_pb* b, int field, kernix_pb* sub)
{
    if (sub->err) b->err = 1;
    pb_bytes(b, field, sub->p, sub->n);
    free(sub->p);
    *sub = (kernix_pb){ 0 };
}

static void pb_empty(kernix_pb* b, int field) { pb_bytes(b, field, NULL, 0); }

static void pb_uints(kernix_pb* b, int field, const uint64_t* v, int n) /* packed */
{
    kernix_pb t = { 0 };
    for (int i = 0; i < n; i++) pb_varint(&t, v[i]);
    pb_msg(b, field, &t);
}

static void pb_weights(kernix_pb* b, int field, const float* w, size_t n)
{
    kernix_pb t = { 0 };
    pb_bytes(&t, 1, w, n * sizeof(float)); /* floatValue: packed float */
    pb_msg(b, field, &t);
}

/* (위·아래, 왼쪽·오른쪽) 패딩 → BorderAmounts */
static void pb_border(kernix_pb* b, int field, int pt, int pb_, int pl, int pr)
{
    kernix_pb ba = { 0 }, e = { 0 };
    pb_uint(&e, 1, (uint64_t)pt); pb_uint(&e, 2, (uint64_t)pb_); pb_msg(&ba, 10, &e);
    pb_uint(&e, 1, (uint64_t)pl); pb_uint(&e, 2, (uint64_t)pr); pb_msg(&ba, 10, &e);
    pb_msg(b, field, &ba);
}

/* 층 하나: 입력 이름 n_in개, 출력 이름 하나, 층 종류 field의 인자 메시지 params (비움) */
static void pb_layer(kernix_pb* net, const char* const* in, int n_in, const char* out, int field, kernix_pb* params)
{
    static int seq;
    char name[32];
    snprintf(name, sizeof name, "l%d", __atomic_add_fetch(&seq, 1, __ATOMIC_RELAXED));
    kernix_pb l = { 0 };
    pb_str(&l, 1, name);
    for (int k = 0; k < n_in; k++) pb_str(&l, 2, in[k]);
    pb_str(&l, 3, out);
    pb_msg(&l, field, params);
    pb_msg(net, 1, &l);
}

static void pb_layer1(kernix_pb* net, const char* in, const char* out, int field, kernix_pb* params)
{
    pb_layer(net, &in, 1, out, field, params);
}

static void pb_transpose(kernix_pb* net, const char* in, const char* out, const uint64_t axes[4])
{
    kernix_pb t = { 0 };
    pb_uints(&t, 1, axes, 4);
    pb_layer1(net, in, out, 985, &t);
}

/* out = act(in). 층 1~2개 (silu·hard_swish는 x × 시그모이드). tmp는 중간 이름용 */
static void pb_act(kernix_pb* net, const char* in, const char* out, int act, float alpha)
{
    kernix_pb p = { 0 }, a = { 0 };
    char tmp[64];
    snprintf(tmp, sizeof tmp, "%s_g", out);
    switch (act) {
    case KERNIX_ACT_RELU: pb_empty(&a, 10); break;
    case KERNIX_ACT_LEAKY_RELU: { kernix_pb q = { 0 }; pb_float(&q, 1, alpha); pb_msg(&a, 15, &q); break; }
    case KERNIX_ACT_TANH: pb_empty(&a, 30); break;
    case KERNIX_ACT_SIGMOID: case KERNIX_ACT_SILU: pb_empty(&a, 40); break;
    case KERNIX_ACT_HARD_SIGMOID: case KERNIX_ACT_HARD_SWISH: { /* relu6(x + 3) / 6 = min(max(x/6 + 1/2, 0), 1) */
        kernix_pb q = { 0 };
        pb_float(&q, 1, 1.0f / 6.0f); pb_float(&q, 2, 0.5f);
        pb_msg(&a, 41, &q);
        break;
    }
    case KERNIX_ACT_ELU: { kernix_pb q = { 0 }; pb_float(&q, 1, alpha); pb_msg(&a, 50, &q); break; }
    case KERNIX_ACT_SOFTSIGN: pb_empty(&a, 60); break;
    case KERNIX_ACT_SOFTPLUS: pb_empty(&a, 70); break;
    case KERNIX_ACT_RELU6:
        pb_float(&p, 1, 0.0f); pb_float(&p, 2, 6.0f);
        pb_layer1(net, in, out, 660, &p);
        return;
    case KERNIX_ACT_GELU: case KERNIX_ACT_GELU_TANH:
        pb_uint(&p, 1, act == KERNIX_ACT_GELU ? 0 : 1);
        pb_layer1(net, in, out, 795, &p);
        return;
    default: /* kernix_coreml_act_ok가 막음 */
        net->err = 1;
        return;
    }
    const int gated = act == KERNIX_ACT_SILU || act == KERNIX_ACT_HARD_SWISH;
    pb_layer1(net, in, gated ? tmp : out, 130, &a);
    if (gated) {
        const char* xs[2] = { in, tmp };
        kernix_pb m = { 0 };
        pb_layer(net, xs, 2, out, 900, &m);
    }
}

/* Core ML 쪽 모양 (C, H, W). 벡터는 C×1×1, 시퀀스 T×C는 C×T×1 */
static void kernix_cm_shape(const kernix_val* v, int* C, int* H, int* W)
{
    *C = v->c;
    *H = v->nd >= 2 ? v->d[0] : 1;
    *W = v->nd == 3 ? v->d[1] : 1;
}

static void pb_feature(kernix_pb* desc, int field, const char* name, int B, int C, int H, int W)
{
    kernix_pb f = { 0 }, t = { 0 }, a = { 0 };
    const uint64_t shape[4] = { (uint64_t)B, (uint64_t)H, (uint64_t)W, (uint64_t)C }; /* KERNIX 버퍼 모양 (배치, H, W, C) */
    pb_uints(&a, 1, shape, 4);
    pb_uint(&a, 2, 65568);
    pb_msg(&t, 5, &a);
    pb_str(&f, 1, name);
    pb_msg(&f, 3, &t);
    pb_msg(desc, field, &f);
}

/* CONV(groups 포함) 또는 LINEAR(1×1)를 convolution 층으로. w: KERNIX 가중치 (K×cout, K = kh·kw·cg 순서 (ky, kx, ci)) */
static void pb_conv(kernix_pb* net, const char* in, const char* out, const kernix_step* s, const float* w, int cin, int cout,
                    int groups, int kh, int kw)
{
    const int cg = cin / groups;
    const size_t n = (size_t)cout * cg * kh * kw;
    float* wc = malloc(sizeof(float) * (n ? n : 1));
    if (wc == NULL) { net->err = 1; return; }
    for (int co = 0; co < cout; co++)
        for (int ci = 0; ci < cg; ci++)
            for (int ky = 0; ky < kh; ky++)
                for (int kx = 0; kx < kw; kx++) /* Core ML (cout, cg, kh, kw) */
                    wc[(((size_t)co * cg + ci) * kh + ky) * kw + kx] = w[(((size_t)ky * kw + kx) * cg + ci) * cout + co];
    const int conv = s->kind == KERNIX_OP_CONV;
    const uint64_t ks[2] = { (uint64_t)kh, (uint64_t)kw };
    const uint64_t st[2] = { conv ? (uint64_t)s->sh : 1, conv ? (uint64_t)s->sw : 1 };
    const uint64_t dl[2] = { conv ? (uint64_t)s->dh : 1, conv ? (uint64_t)s->dw : 1 };
    kernix_pb p = { 0 }, v = { 0 };
    pb_uint(&p, 1, (uint64_t)cout);
    pb_uint(&p, 2, (uint64_t)cg);
    pb_uint(&p, 10, (uint64_t)groups);
    pb_uints(&p, 20, ks, 2);
    pb_uints(&p, 30, st, 2);
    pb_uints(&p, 40, dl, 2);
    pb_border(&v, 1, conv ? s->pt : 0, conv ? s->pb : 0, conv ? s->pl : 0, conv ? s->pr : 0);
    pb_msg(&p, 50, &v);
    pb_uint(&p, 70, s->bias != NULL);
    pb_weights(&p, 90, wc, n);
    if (s->bias != NULL) pb_weights(&p, 91, s->bias, (size_t)cout);
    free(wc);
    char pre[64];
    snprintf(pre, sizeof pre, "%s_c", out);
    pb_layer1(net, in, s->act == KERNIX_ACT_NONE ? out : pre, 100, &p);
    if (s->act != KERNIX_ACT_NONE) pb_act(net, pre, out, s->act, s->alpha);
}

int kernix_coreml_write_spec(const kernix_model* m, int seg, int batch, const char* path)
{
    if (m == NULL || path == NULL || seg < 0 || seg >= m->n_seg || batch < 1) return -1;
    const kernix_ane_seg* g = &m->seg[seg];
    const int nv = m->n_vals;
    int* root = malloc(sizeof(int) * (size_t)nv); /* 별칭(Reshape·Permute)을 따라간 실제 값 번호 */
    if (root == NULL) return -1;
    for (int v = 0; v < nv; v++) root[v] = v;
    kernix_pb net = { 0 };
    char a[64], b[64], o[64];
#define VNAME(buf, v) snprintf(buf, sizeof buf, "v%d", root[v])
    static const uint64_t to_nchw[4] = { 0, 3, 1, 2 }, to_nhwc[4] = { 0, 2, 3, 1 };
    VNAME(a, g->in_v);
    pb_transpose(&net, "x", a, to_nchw);
    for (int i = g->first; i <= g->last && !net.err; i++) {
        const kernix_step* s = &m->steps[i];
        const int ov = i + 1;
        if (kernix_ane_is_view(s->kind)) { root[ov] = root[s->inp[0]]; continue; }
        VNAME(a, s->inp[0]);
        VNAME(o, ov);
        kernix_pb p = { 0 };
        switch (s->kind) {
        case KERNIX_OP_CONV:
            pb_conv(&net, a, o, s, s->w->tensor, s->in, s->out, s->groups, s->kh, s->kw);
            break;
        case KERNIX_OP_LINEAR: {
            float* w = s->w != NULL ? s->w->tensor : NULL;
            float* deq = NULL;
            if (w == NULL) { /* int8: float로 풀어 넘김 */
                if ((deq = malloc(sizeof(float) * (size_t)s->q->k * (size_t)s->q->n)) == NULL) { net.err = 1; break; }
                kernix_qweight_unpack(s->q, deq);
                w = deq;
            }
            pb_conv(&net, a, o, s, w, s->in, s->out, 1, 1, 1);
            free(deq);
            break;
        }
        case KERNIX_OP_ACT:
            pb_act(&net, a, o, s->act, s->alpha);
            break;
        case KERNIX_OP_POOL:
        case KERNIX_OP_GLOBAL_POOL: {
            const int glob = s->kind == KERNIX_OP_GLOBAL_POOL;
            const uint64_t ks[2] = { glob ? 1 : (uint64_t)s->kh, glob ? 1 : (uint64_t)s->kw };
            const uint64_t st[2] = { glob ? 1 : (uint64_t)s->sh, glob ? 1 : (uint64_t)s->sw };
            kernix_pb v = { 0 };
            pb_uint(&p, 1, s->ptype == KERNIX_POOL_MAX ? 0 : 1);
            pb_uints(&p, 10, ks, 2);
            pb_uints(&p, 20, st, 2);
            pb_border(&v, 1, glob ? 0 : s->pt, glob ? 0 : s->pb, glob ? 0 : s->pl, glob ? 0 : s->pr);
            pb_msg(&p, 30, &v);
            pb_uint(&p, 50, !glob && !s->count_pad); /* KERNIX 최댓값 풀링은 패딩 칸을 건너뜀(-∞) = Core ML 기본 */
            pb_uint(&p, 60, glob);
            pb_layer1(&net, a, o, 120, &p);
            break;
        }
        case KERNIX_OP_AFFINE: {
            const uint64_t c = (uint64_t)s->in;
            pb_uints(&p, 1, &c, 1);
            pb_weights(&p, 2, s->gamma, (size_t)s->in);
            pb_uint(&p, 3, 1);
            pb_uints(&p, 4, &c, 1);
            pb_weights(&p, 5, s->beta, (size_t)s->in);
            pb_layer1(&net, a, o, 245, &p);
            break;
        }
        case KERNIX_OP_BINARY: {
            static const int field[] = { [KERNIX_BIN_ADD] = 880, [KERNIX_BIN_SUB] = 905, [KERNIX_BIN_MUL] = 900,
                                         [KERNIX_BIN_MAX] = 875, [KERNIX_BIN_MIN] = 870 };
            VNAME(b, s->inp[1]);
            const char* xs[2] = { a, b }; /* KERNIX sub = 첫 입력 − 둘째 입력 = Core ML subtractBroadcastable */
            pb_layer(&net, xs, 2, o, field[s->bop], &p);
            break;
        }
        case KERNIX_OP_PAD: {
            kernix_pb c = { 0 };
            pb_float(&c, 1, 0.0f);
            pb_msg(&p, 1, &c);
            pb_border(&p, 10, s->pt, s->pb, s->pl, s->pr);
            pb_layer1(&net, a, o, 200, &p);
            break;
        }
        case KERNIX_OP_UPSAMPLE: {
            const uint64_t sf[2] = { (uint64_t)s->sh, (uint64_t)s->sw };
            pb_uints(&p, 1, sf, 2);
            pb_uint(&p, 5, 0);
            pb_layer1(&net, a, o, 210, &p);
            break;
        }
        case KERNIX_OP_CONCAT: { /* 채널 축 */
            char names[KERNIX_MAX_INPUTS][64];
            const char* xs[KERNIX_MAX_INPUTS];
            for (int k = 0; k < s->n_in; k++) { VNAME(names[k], s->inp[k]); xs[k] = names[k]; }
            pb_layer(&net, xs, s->n_in, o, 320, &p);
            break;
        }
        default:
            net.err = 1;
            break;
        }
    }
    VNAME(o, g->out_v);
    pb_transpose(&net, o, "y", to_nhwc);
#undef VNAME
    pb_uint(&net, 5, 1); /* EXACT_ARRAY_MAPPING */

    int ci, hi, wi, co, ho, wo;
    kernix_cm_shape(&m->vals[g->in_v], &ci, &hi, &wi);
    kernix_cm_shape(&m->vals[root[g->out_v]], &co, &ho, &wo); /* 별칭 출력은 원래 값의 모양 (메모리는 같음) */
    free(root);
    kernix_pb desc = { 0 }, model = { 0 };
    pb_feature(&desc, 1, "x", batch, ci, hi, wi);
    pb_feature(&desc, 10, "y", batch, co, ho, wo);
    pb_uint(&model, 1, 4); /* iOS 13 / macOS 10.15: transpose·broadcastable·gelu·clip */
    pb_msg(&model, 2, &desc);
    pb_msg(&model, 500, &net);
    int rc = -1;
    if (!model.err) {
        FILE* f = fopen(path, "wb");
        if (f != NULL) {
            rc = fwrite(model.p, 1, model.n, f) == model.n ? 0 : -1;
            if (fclose(f) != 0) rc = -1;
        }
    }
    free(model.p);
    return rc;
}

/* ---- 실행: Core ML 모델 하나 (고정 배치) ---- */
@interface KernixCoreMLRunner : NSObject
@property(nonatomic, strong) MLModel* model;
@property(nonatomic, strong) NSArray<NSNumber*>* inShape;
@property(nonatomic, strong) NSArray<NSNumber*>* inStrides;
@property(nonatomic, strong) NSArray<NSNumber*>* outShape;
@property(nonatomic, strong) NSArray<NSNumber*>* outStrides;
@property(nonatomic) size_t outCount;
@end
@implementation KernixCoreMLRunner
@end

static NSArray<NSNumber*>* kernix_strides(NSArray<NSNumber*>* shape)
{
    NSMutableArray<NSNumber*>* s = [NSMutableArray arrayWithCapacity:shape.count];
    NSInteger acc = 1;
    for (NSInteger k = (NSInteger)shape.count - 1; k >= 0; k--) {
        [s insertObject:@(acc) atIndex:0];
        acc *= shape[(NSUInteger)k].integerValue;
    }
    return s;
}

/* 구간 seg를 batch 표본 고정으로 컴파일해 불러옴. 실패 시 nil. 임시 파일은 TMPDIR(없으면 NSTemporaryDirectory) 아래에 쓰고 지움 */
static KernixCoreMLRunner* kernix_coreml_build(const kernix_model* m, int seg, int batch)
{
    if (@available(macOS 13.0, *)) {
        const char* t = getenv("TMPDIR");
        NSString* dir = t != NULL && *t ? @(t) : NSTemporaryDirectory();
        NSString* name = [NSString stringWithFormat:@"kernix_%d_%d_%@.mlmodel", getpid(), seg, NSUUID.UUID.UUIDString];
        NSURL* spec = [NSURL fileURLWithPath:[dir stringByAppendingPathComponent:name]];
        if (kernix_coreml_write_spec(m, seg, batch, spec.fileSystemRepresentation) != 0) return nil;
        NSError* err = nil;
        NSURL* compiled = [MLModel compileModelAtURL:spec error:&err];
        [NSFileManager.defaultManager removeItemAtURL:spec error:nil];
        if (compiled == nil) return nil;
        MLModelConfiguration* cfg = [MLModelConfiguration new];
        cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        MLModel* model = [MLModel modelWithContentsOfURL:compiled configuration:cfg error:&err];
        [NSFileManager.defaultManager removeItemAtURL:compiled error:nil]; /* 불러온 뒤에는 필요 없음 (매핑된 파일은 해제 때까지 남음) */
        if (model == nil) return nil;
        KernixCoreMLRunner* r = [KernixCoreMLRunner new];
        r.model = model;
        r.inShape = model.modelDescription.inputDescriptionsByName[@"x"].multiArrayConstraint.shape;
        r.outShape = model.modelDescription.outputDescriptionsByName[@"y"].multiArrayConstraint.shape;
        if (r.inShape.count != 4 || r.outShape.count != 4) return nil;
        r.inStrides = kernix_strides(r.inShape);
        r.outStrides = kernix_strides(r.outShape);
        r.outCount = (size_t)(r.outShape[0].integerValue * r.outStrides[0].integerValue);
        return r;
    }
    return nil;
}

int kernix_coreml_predict(void* h, const float* x, float* y)
{
    if (@available(macOS 13.0, *)) {
        @autoreleasepool { /* C에서 반복 호출: 호출마다 임시 객체를 비움 */
            KernixCoreMLRunner* r = (__bridge KernixCoreMLRunner*)h;
            NSError* err = nil;
            MLMultiArray* in = [[MLMultiArray alloc] initWithDataPointer:(void*)x shape:r.inShape dataType:MLMultiArrayDataTypeFloat32
                                                                 strides:r.inStrides deallocator:nil error:&err];
            MLMultiArray* out = [[MLMultiArray alloc] initWithDataPointer:y shape:r.outShape dataType:MLMultiArrayDataTypeFloat32
                                                                  strides:r.outStrides deallocator:nil error:&err];
            if (in == nil || out == nil) return -1;
            MLDictionaryFeatureProvider* fp = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{ @"x" : in } error:&err];
            if (fp == nil) return -1;
            MLPredictionOptions* opt = [MLPredictionOptions new];
            opt.outputBackings = @{ @"y" : out }; /* 결과를 KERNIX 출력 버퍼에 바로 */
            id<MLFeatureProvider> res = [r.model predictionFromFeatures:fp options:opt error:&err];
            if (res == nil) return -1;
            MLMultiArray* got = [res featureValueForName:@"y"].multiArrayValue;
            if (got == nil) return -1;
            if (got.dataPointer != (void*)y) { /* 버퍼를 쓰지 않은 경우: 연속 float32면 복사 */
                if (got.dataType != MLMultiArrayDataTypeFloat32 || (size_t)got.count != r.outCount) return -1;
                __block int ok = 0;
                [got getBytesWithHandler:^(const void* bytes, NSInteger size) {
                    if ((size_t)size >= r.outCount * sizeof(float)) { memcpy(y, bytes, r.outCount * sizeof(float)); ok = 1; }
                }];
                if (!ok) return -1;
            }
            return 0;
        }
    }
    return -1;
}

void kernix_coreml_release(void* h)
{
    if (h != NULL) CFBridgingRelease(h);
}

/* ---- 뒤 컴파일 큐: 모델당 직렬 큐 하나, 구간마다 결과 칸 하나 ---- */
@interface KernixCoreMLCtx : NSObject {
@public
    dispatch_queue_t queue;
    os_unfair_lock lock;
    int n;
    int* pending;   /* 컴파일 중인 표본 수, 0 = 없음 */
    void** ready;   /* 끝난 모델 (CFBridgingRetain), poll이 가져감 */
    int* ready_b;
    int* failed;
}
@end
@implementation KernixCoreMLCtx
- (void)dealloc
{
    for (int k = 0; k < n; k++) kernix_coreml_release(ready[k]);
    free(pending); free(ready); free(ready_b); free(failed);
}
@end

void* kernix_coreml_ctx_new(int n_seg)
{
    KernixCoreMLCtx* c = [KernixCoreMLCtx new];
    c->queue = dispatch_queue_create("kernix.coreml.compile", DISPATCH_QUEUE_SERIAL);
    c->lock = OS_UNFAIR_LOCK_INIT;
    c->n = n_seg;
    c->pending = calloc((size_t)n_seg, sizeof(int));
    c->ready = calloc((size_t)n_seg, sizeof(void*));
    c->ready_b = calloc((size_t)n_seg, sizeof(int));
    c->failed = calloc((size_t)n_seg, sizeof(int));
    if (c->pending == NULL || c->ready == NULL || c->ready_b == NULL || c->failed == NULL) return NULL;
    return (void*)CFBridgingRetain(c);
}

void kernix_coreml_ctx_free(void* ctx)
{
    if (ctx == NULL) return;
    KernixCoreMLCtx* c = (__bridge KernixCoreMLCtx*)ctx;
    dispatch_sync(c->queue, ^{}); /* 직렬 큐: 돌고 있는 컴파일이 끝나기를 기다림 */
    CFBridgingRelease(ctx);
}

int kernix_coreml_request(void* ctx, const kernix_model* m, int seg, int batch)
{
    KernixCoreMLCtx* c = (__bridge KernixCoreMLCtx*)ctx;
    if (seg < 0 || seg >= c->n || batch < 1) return 0;
    os_unfair_lock_lock(&c->lock);
    const int busy = c->pending[seg] != 0;
    if (!busy) c->pending[seg] = batch;
    os_unfair_lock_unlock(&c->lock);
    if (busy) return 0;
    dispatch_async(c->queue, ^{
        @autoreleasepool {
            KernixCoreMLRunner* r = kernix_coreml_build(m, seg, batch); /* m의 가중치·모양은 로드 뒤 바뀌지 않음, free가 큐를 기다림 */
            os_unfair_lock_lock(&c->lock);
            c->pending[seg] = 0;
            if (r != nil) {
                kernix_coreml_release(c->ready[seg]); /* 가져가지 않은 이전 결과 */
                c->ready[seg] = (void*)CFBridgingRetain(r);
                c->ready_b[seg] = batch;
            } else {
                c->failed[seg] = 1;
            }
            os_unfair_lock_unlock(&c->lock);
        }
    });
    return 1;
}

int kernix_coreml_poll(void* ctx, int seg, void** h, int* batch)
{
    KernixCoreMLCtx* c = (__bridge KernixCoreMLCtx*)ctx;
    if (seg < 0 || seg >= c->n) return 0;
    int r = 0;
    os_unfair_lock_lock(&c->lock);
    if (c->failed[seg]) {
        c->failed[seg] = 0;
        r = -1;
    } else if (c->ready[seg] != NULL) {
        *h = c->ready[seg];
        *batch = c->ready_b[seg];
        c->ready[seg] = NULL;
        r = 1;
    }
    os_unfair_lock_unlock(&c->lock);
    return r;
}
