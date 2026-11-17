/* .knx 로더 퍼징 하네스: 파일마다 로드하고, 받아들여지면 무작위 입력으로 forward까지 실행한다.
 * ASan/UBSan으로 빌드해 손상된 파일이 메모리 오류를 내지 않는지 본다 (test/knx_fuzz_test.py가 파일을 만들고 부름) */
#include "kernix_model.h"

int main(int argc, char** argv)
{
    int loaded = 0;
    for (int i = 1; i < argc; i++) {
        kernix_model* m = kernix_model_load(argv[i]);
        if (m == NULL) continue;
        loaded++;
        const int rows = 3 * m->seq_len;
        float_tensor* x = float_zeros(rows, m->in_dim);
        if (x != NULL) {
            for (size_t k = 0; k < (size_t)rows * m->in_dim; k++) x->tensor[k] = (float)((k * 2654435761u) % 7) - 3.0f;
            kernix_model_forward(m, x);
            kernix_model_set_workspace_limit(m, 1); /* 표본 하나씩 나눠 실행하는 경로도 */
            kernix_model_forward(m, x);
            float_free(x);
        }
        kernix_model_free(m);
    }
    printf("loaded %d / %d\n", loaded, argc - 1);
    return 0;
}
