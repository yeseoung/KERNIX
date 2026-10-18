#include "kernix.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define TASK_SIZE 4

typedef struct {
    float_tensor* tensor;
    int start;
    int end;
    float sum;
} task_args;

void* work(void* arg) {
    task_args* task = (task_args*)arg;

    task->sum = 0;

    for (int i = task->start; i < task->end; i++) {
        task->sum += task->tensor->tensor[i];
    }

    return NULL;
}

int main(void) {
    float_tensor* train_x = float_zeros(2, 2);

    train_x->tensor[0] = 1;
    train_x->tensor[1] = 2;
    train_x->tensor[2] = 3;
    train_x->tensor[3] = 4;

    show_floats(train_x);

    int* p = task_balancer(train_x);
    if (p == NULL) {
        float_free(train_x);
        return 1;
    }

    pthread_t threads[TASK_SIZE];
    task_args tasks[TASK_SIZE];

    for (int i = 0; i < TASK_SIZE; i++) {
        tasks[i].tensor = train_x;
        tasks[i].start = (i == 0) ? 0 : p[i - 1];
        tasks[i].end = p[i];
        tasks[i].sum = 0;

        if (pthread_create(&threads[i], NULL, work, &tasks[i]) != 0) {
            fprintf(stderr, "스레드 생성 실패\n");
            free(p);
            float_free(train_x);
            return 1;
        }
    }

    float total = 0;

    for (int i = 0; i < TASK_SIZE; i++) {
        pthread_join(threads[i], NULL);
        total += tasks[i].sum;
    }

    printf("합계: %.1f\n", total);

    free(p);
    float_free(train_x);

    return 0;
}