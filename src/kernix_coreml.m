/* ANE 보조: .knx 그래프의 구간을 Core ML NeuralNetwork 모델로 만들어 뉴럴엔진에서 실행 (-DKERNIX_USE_COREML).
 * 설계: docs/superpowers/specs/2026-11-16-ane-assist-design.md. 구간 찾기와 경로 선택은 include/kernix_model.h */
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
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
