/* 스레드 풀(src/kernix_runtime.c) 검사: 조각 경계, 직렬/병렬 경계, 동시 호출, 종료 후 재사용 */
#include <assert.h>
#include "kernix.h"

static double naive_sum(const float_tensor* t)
{
    double s = 0.0;
    for (size_t i = 0; i < (size_t)t->row * t->col; i++) s += t->tensor[i];
    return s;
}

static void check_sum(int rows, int cols)
{
    float_tensor* t = float_zeros(rows, cols);
    assert(t);
    for (size_t i = 0; i < (size_t)rows * cols; i++) t->tensor[i] = (float)((i * 2654435761u) % 1000) / 1000.0f;
    double got = -1.0;
    assert(kernix_thread_pool_sum(t, &got) == 0);
    double want = naive_sum(t);
    if (fabs(got - want) > 1e-6 * (fabs(want) + 1)) {
        fprintf(stderr, "합산 %d×%d: %.6f != %.6f\n", rows, cols, got, want);
        exit(1);
    }
    float_free(t);
}

static void check_matmul(int m, int k, int n)
{
    float_tensor *a = float_zeros(m, k), *b = float_zeros(k, n);
    int_tensor *ia = int_zeros(m, k), *ib = int_zeros(k, n);
    for (int i = 0; i < m * k; i++) { a->tensor[i] = (float)(i % 7) - 3; ia->tensor[i] = i % 7 - 3; }
    for (int i = 0; i < k * n; i++) { b->tensor[i] = (float)(i % 5) - 2; ib->tensor[i] = i % 5 - 2; }
    float_tensor* c = float_matmul_native_parallel(a, b);
    int_tensor* ic = int_matmul_parallel(ia, ib);
    assert(c && ic);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            long s = 0;
            for (int x = 0; x < k; x++) s += (long)ia->tensor[i * k + x] * ib->tensor[x * n + j];
            if (c->tensor[i * n + j] != (float)s || ic->tensor[i * n + j] != s) {
                fprintf(stderr, "행렬곱 %d×%d×%d [%d,%d]: float %.0f, int %d, 기대 %ld\n", m, k, n, i, j,
                        c->tensor[i * n + j], ic->tensor[i * n + j], s);
                exit(1);
            }
        }
    float_free(a); float_free(b); float_free(c); int_free(ia); int_free(ib); int_free(ic);
}

static void* hammer(void* arg)
{
    (void)arg;
    for (int r = 0; r < 50; r++) check_sum(37, 9973); /* 병렬 경로 (조각 6개) */
    return NULL;
}

int main(void)
{
    /* 합산: 빈 텐서, 조각 1개(직렬), 조각 경계 ±1, 여러 조각 */
    int sums[][2] = { {0, 0}, {0, 5}, {1, 1}, {1, 7}, {1, 65535}, {1, 65536}, {1, 65537}, {3, 65536}, {1031, 1017} };
    for (size_t i = 0; i < sizeof sums / sizeof *sums; i++) check_sum(sums[i][0], sums[i][1]);
    /* 행렬곱: 빈 결과, 직렬(작은), 병렬, 행 수가 조각 수(64)보다 적음/많음/나누어떨어지지 않음 */
    int mms[][3] = { {0, 4, 4}, {3, 0, 5}, {1, 1, 1}, {5, 7, 3}, {33, 129, 257}, {127, 128, 129}, {200, 64, 70} };
    for (size_t i = 0; i < sizeof mms / sizeof *mms; i++) check_matmul(mms[i][0], mms[i][1], mms[i][2]);

    /* 여러 스레드가 동시에 풀을 씀: 차례로 실행돼 결과가 섞이지 않아야 함 */
    pthread_t th[4];
    for (int i = 0; i < 4; i++) assert(pthread_create(&th[i], NULL, hammer, NULL) == 0);
    for (int i = 0; i < 4; i++) pthread_join(th[i], NULL);

    /* 종료 후 다시 쓰면 작업자를 새로 만든다 */
    kernix_thread_pool_shutdown();
    check_sum(1031, 1017);
    check_matmul(127, 128, 129);
    kernix_thread_pool_shutdown();
    kernix_thread_pool_shutdown(); /* 두 번 불러도 안전 */

    double s;
    assert(kernix_thread_pool_sum(NULL, &s) == -1);
    assert(kernix_pool_run(3, NULL, NULL) == -1);
    printf("스레드 풀 검사 통과 ✓\n");
    return 0;
}
