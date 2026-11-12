/* kernix_model.h(헤더 전용, static inline)를 공유 라이브러리 심볼로 내보내는 C API. python/kernix.py가 ctypes로 부름.
 *   make python-lib  →  python/libkernix.dylib */
#include "kernix_model.h"

#define KERNIX_API __attribute__((visibility("default")))

KERNIX_API kernix_model* kernix_capi_load(const char* path) { return kernix_model_load(path); }

KERNIX_API void kernix_capi_free(kernix_model* m) { kernix_model_free(m); }

/* 작업 버퍼 해제 (가중치 유지). 다음 forward가 다시 할당 */
KERNIX_API void kernix_capi_trim(kernix_model* m) { kernix_model_trim(m); }

/* 작업 버퍼 상한 (바이트, 0 = 제한 없음). 넘는 배치는 나눠 실행 */
KERNIX_API void kernix_capi_set_workspace_limit(kernix_model* m, size_t bytes) { kernix_model_set_workspace_limit(m, bytes); }

KERNIX_API void kernix_capi_dims(const kernix_model* m, int* in_dim, int* out_dim, int* seq_len)
{
    *in_dim = m->in_dim; *out_dim = m->out_dim; *seq_len = m->seq_len;
}

/* x: rows × in_dim (rows는 seq_len의 배수), out: (rows / seq_len) × out_dim 에 복사. 성공 0, 실패 -1.
 * 출력 버퍼는 모델 소유라 다음 호출이 덮어쓰므로 호출자 메모리로 복사해 돌려줌 */
KERNIX_API int kernix_capi_forward(kernix_model* m, const float* x, int rows, float* out)
{
    if (m == NULL || x == NULL || out == NULL || rows <= 0) return -1;
    const float_tensor in = { rows, m->in_dim, (float*)x }; /* forward는 입력을 바꾸지 않음 */
    const float_tensor* y = kernix_model_forward(m, &in);
    if (y == NULL) return -1;
    memcpy(out, y->tensor, sizeof(float) * (size_t)y->row * (size_t)y->col);
    return 0;
}

/* 표본 하나의 입력·출력 모양 (channels-last, 축 수 ≤ 4)과 입력 layout (0 = channels-last, 1 = PyTorch channels-first) */
KERNIX_API void kernix_capi_shapes(const kernix_model* m, int* in_nd, int* in_shape, int* out_nd, int* out_shape, int* layout)
{
    *in_nd = m->in_nd; *out_nd = m->out_nd; *layout = m->layout;
    for (int k = 0; k < KERNIX_MAX_DIMS; k++) { in_shape[k] = m->in_shape[k]; out_shape[k] = m->out_shape[k]; }
}
