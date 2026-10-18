#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "kernix.h"

int main(void) {
    // 1000 x 1000 = 1,000,000개의 원소
    int rows = 10000;
    int cols = 10000;

    float_tensor* train_x = float_zeros(rows, cols);

    // 테스트 데이터 초기화
    for (int i = 0; i < rows * cols; i++) {
        train_x->tensor[i] = 1.0f;
    }

    // 순차 연산 시작
    clock_t start = clock();

    double sum = 0.0f;
    int data_size = train_x->row * train_x->col;

    for (int i = 0; i < data_size; i++) {
        sum += train_x->tensor[i];
    }

    // 순차 연산 종료
    clock_t end = clock();

    double elapsed_time =
        (double)(end - start) / CLOCKS_PER_SEC;

    printf("원소 개수: %d\n", data_size);
    printf("합계: %.1f\n", sum);
    printf("순차 연산 시간: %.6f초\n", elapsed_time);

    float_free(train_x);

    return 0;
}