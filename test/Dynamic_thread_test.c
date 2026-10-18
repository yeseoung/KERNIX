#include "kernix.h"
#include <time.h>

#define ROWS 10000
#define COLS 10000
#define TEST_RUNS 20

int main(void)
{
    float_tensor* a = float_zeros(ROWS, COLS);

    if (a == NULL || a->tensor == NULL) {
        fprintf(stderr, "텐서 메모리 할당 실패\n");
        return 1;
    }

    size_t element_count = (size_t)ROWS * COLS;

    // 데이터 초기화: 측정 시간에서 제외
    for (size_t i = 0; i < element_count; i++) {
        a->tensor[i] = 1.0f;
    }

    double sum = 0.0;

    // 워밍업 1회: 스레드 풀 초기화 비용을 측정에서 제외
    if (kernix_thread_pool_sum(a, &sum) != 0) {
        fprintf(stderr, "워밍업 실패\n");
        kernix_thread_pool_shutdown();
        float_free(a);
        return 1;
    }

    struct timespec start, end;
    double times[TEST_RUNS];
    double total_time = 0.0;

    // 동일한 스레드 풀을 재사용하며 4회 측정
    for (int i = 0; i < TEST_RUNS; i++) {
        clock_gettime(CLOCK_MONOTONIC, &start);

        int rc = kernix_thread_pool_sum(a, &sum);

        clock_gettime(CLOCK_MONOTONIC, &end);

        if (rc != 0) {
            fprintf(stderr, "합산 실패\n");
            kernix_thread_pool_shutdown();
            float_free(a);
            return 1;
        }

        times[i] =
            (double)(end.tv_sec - start.tv_sec)
            + (double)(end.tv_nsec - start.tv_nsec) / 1000000000.0;

        total_time += times[i];
    }

    // 측정이 끝난 뒤 출력
    printf("합계: %.1f\n", sum);
    printf("원소 개수: %zu\n", element_count);

    printf("\n--- 실행 시간 ---\n");

    for (int i = 0; i < TEST_RUNS; i++) {
        printf("%d회차: %.6f초\n", i + 1, times[i]);
    }

    // 평균
    double average = total_time / TEST_RUNS;

    // 중앙값 계산을 위해 4개 값을 오름차순 정렬
    for (int i = 0; i < TEST_RUNS - 1; i++) {
        for (int j = i + 1; j < TEST_RUNS; j++) {
            if (times[i] > times[j]) {
                double temp = times[i];
                times[i] = times[j];
                times[j] = temp;
            }
        }
    }

    double median =
    (times[TEST_RUNS / 2 - 1] +
     times[TEST_RUNS / 2]) / 2.0;

    printf("\n평균 실행 시간: %.6f초\n", average);
    printf("중앙값 실행 시간: %.6f초\n", median);

    kernix_thread_pool_shutdown();
    float_free(a);

    return 0;
}