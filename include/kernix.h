
#ifndef KERNIX_H
#define KERNIX_H

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stddef.h>

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

typedef struct {
    float_tensor* tensor;
    size_t start;
    size_t end;
    double sum;
} task_args;

static void kernix_sum_work(task_args* task)
{
    const float* data = task->tensor->tensor;

    float sum[8] = {0.0f};

    size_t i = task->start;
    const size_t end = task->end;

    for (; end - i >= 8; i += 8) {
        for (size_t j = 0; j < 8; j++) {
            sum[j] += data[i + j];
        }
    }

    for (; i < end; i++) {
        sum[0] += data[i];
    }

    double total = 0.0;

    for (size_t j = 0; j < 8; j++) {
        total += sum[j];
    }

    task->sum = total;
}

static inline int_tensor* int_zeros(int rows,int cols)
{
    int_tensor* p = malloc(sizeof(int_tensor));

    p->row = rows;
    p->col = cols;
    p->tensor = malloc(sizeof(*p->tensor)*rows*cols);

    memset(p->tensor,0,sizeof(int)*rows*cols);

    return p;

}
static inline float_tensor* float_zeros(int rows,int cols)
{
    float_tensor* p = malloc(sizeof(float_tensor));
    
    p->row = rows;
    p->col = cols;
    p->tensor = malloc(sizeof(*p->tensor)*rows*cols);

    memset(p->tensor,0.0,sizeof(*p->tensor)*rows*cols);

    return p;
}
static inline uint8_tensor* uint8_zeros(int rows,int cols)
{
    uint8_tensor* p = malloc(sizeof(uint8_tensor));

    p->row = rows;
    p->col = cols;
    p->tensor = malloc(sizeof(*p->tensor)*rows*cols);

    memset(p->tensor,0,sizeof(*p->tensor)*rows*cols);

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
    free(t->tensor);
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
static inline void show_in8ts_impl(uint8_tensor* tensor,const char *name) {

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

static inline int int_dot(const int_tensor* a,const int_tensor* b){
    int sum = 0;
    
    int a_size = a -> row * a -> col;
    int b_size = b -> row * b-> col;

    if (a_size != b_size) {
        return -1;
    }

    for(int i = 0; i < a_size;i++){
        sum += a->tensor[i] * b->tensor[i];
    }
    return sum;
}
static inline float float_dot(const float_tensor* a,const float_tensor* b){
    float sum = 0;
    
    float a_size = a -> row * a -> col;
    float b_size = b -> row * b-> col;

    if (a_size != b_size) {
        return -1;
    }

    for(int i = 0; i < a_size;i++){
        sum += a->tensor[i] * b->tensor[i];
    }
    return sum;
}
static inline int uint8_dot(const uint8_tensor* a,const uint8_tensor* b){
    int sum = 0;
    
    uint8_t a_size = a -> row * a -> col;
    uint8_t b_size = b -> row * b-> col;

    if (a_size != b_size) {
        return -1;
    }

    for(int i = 0; i < a_size;i++){
        sum += a->tensor[i] * b->tensor[i];
    }
    return sum;
}

static inline int_tensor* ints_add(const int_tensor* a, const int_tensor* b){

    int_tensor* p = malloc(sizeof(int_tensor));
    p->row = a->row;
    p->col = a->col;
    int a_size = a->col * a->row;
    p->tensor = malloc(sizeof(*p->tensor)*a->col*a->row);

    for(int i = 0;i < a_size;i++){
        p->tensor[i] = a->tensor[i] + b->tensor[i];
    }
    return p;

}
static inline float_tensor* floats_add(const float_tensor* a, const float_tensor* b){

    float_tensor* p = malloc(sizeof(float_tensor));
    p->row = a->row;
    p->col = a->col;
    int a_size = a->col * a->row;
    p->tensor = malloc(sizeof(*p->tensor)*a->col*a->row);

    for(int i = 0;i < a_size;i++){
        p->tensor[i] = a->tensor[i] + b->tensor[i];
    }
    return p;

}

static inline int_tensor* int_matmul(const int_tensor* a,const int_tensor* b){
    int_tensor* p = int_zeros(a->row,b->col);

    for (int i = 0; i < a->row; i++) {
        for (int j = 0; j < b->col; j++) {
            int sum = 0;

            for (int k = 0; k < a->col; k++) {
                int a_value = a->tensor[i * a->col + k];
                int b_value = b->tensor[k * b->col + j];

                sum += a_value * b_value;
            }

            p->tensor[i * p->col + j] = sum;
        }
    }

    return p;
}
static inline float_tensor* float_matmul(const float_tensor* a,const float_tensor* b){
    float_tensor* p = float_zeros(a->row,b->col);
    
    for (int i = 0; i < a->row; i++) {
        for (int j = 0; j < b->col; j++) {
            float sum = 0;

            for (int k = 0; k < a->col; k++) {
                float a_value = a->tensor[i * a->col + k];
                float b_value = b->tensor[k * b->col + j];

                sum += a_value * b_value;
            }

            p->tensor[i * p->col + j] = sum;
        }
    }

    return p;
}

#define TASK_SIZE 8

typedef struct {
    pthread_t threads[TASK_SIZE];
    task_args tasks[TASK_SIZE];

    pthread_mutex_t mutex;
    pthread_cond_t work_cond;
    pthread_cond_t done_cond;

    size_t generation;
    int completed;
    int stopping;
    int initialized;
    int failed;
} kernix_thread_pool;

static kernix_thread_pool kernix_pool = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .work_cond = PTHREAD_COND_INITIALIZER,
    .done_cond = PTHREAD_COND_INITIALIZER
};

/* 작업자 스레드 */
static void* kernix_worker(void* arg)
{
    size_t worker_id = (size_t)(uintptr_t)arg;
    size_t last_generation = 0;

    pthread_mutex_lock(&kernix_pool.mutex);

    for (;;) {
        while (!kernix_pool.stopping &&
               kernix_pool.generation == last_generation) {
            pthread_cond_wait(
                &kernix_pool.work_cond,
                &kernix_pool.mutex
            );
        }

        if (kernix_pool.stopping) {
            pthread_mutex_unlock(&kernix_pool.mutex);
            return NULL;
        }

        last_generation = kernix_pool.generation;

        task_args* task = &kernix_pool.tasks[worker_id];

        pthread_mutex_unlock(&kernix_pool.mutex);

        /* 실제 합산 작업 */
        kernix_sum_work(task);

        pthread_mutex_lock(&kernix_pool.mutex);

        kernix_pool.completed++;

        if (kernix_pool.completed == TASK_SIZE) {
            pthread_cond_signal(&kernix_pool.done_cond);
        }
    }
}

/* 스레드 풀 초기화: 작업자 8개를 한 번만 생성 */
static int kernix_thread_pool_init(void)
{
    if (kernix_pool.initialized) {
        return 0;
    }

    if (kernix_pool.failed || kernix_pool.stopping) {
        return -1;
    }

    for (size_t i = 0; i < TASK_SIZE; i++) {
        int rc = pthread_create(
            &kernix_pool.threads[i],
            NULL,
            kernix_worker,
            (void*)(uintptr_t)i
        );

        if (rc != 0) {
            pthread_mutex_lock(&kernix_pool.mutex);
            kernix_pool.stopping = 1;
            pthread_cond_broadcast(&kernix_pool.work_cond);
            pthread_mutex_unlock(&kernix_pool.mutex);

            for (size_t j = 0; j < i; j++) {
                pthread_join(kernix_pool.threads[j], NULL);
            }

            kernix_pool.failed = 1;
            fprintf(stderr, "스레드 풀 생성 실패\n");
            return -1;
        }
    }

    kernix_pool.initialized = 1;
    return 0;
}

/* 작업 제출 및 완료 대기 */
static int kernix_thread_pool_sum(float_tensor* a, double* result)
{
    if (a == NULL || a->tensor == NULL || result == NULL ||
        a->row < 0 || a->col < 0) {
        return -1;
    }

    if (kernix_thread_pool_init() != 0) {
        return -1;
    }

    size_t data_size = (size_t)a->row * (size_t)a->col;

    pthread_mutex_lock(&kernix_pool.mutex);

    if (kernix_pool.stopping) {
        pthread_mutex_unlock(&kernix_pool.mutex);
        return -1;
    }

    for (size_t i = 0; i < TASK_SIZE; i++) {
        kernix_pool.tasks[i].tensor = a;
        kernix_pool.tasks[i].start =
            i * data_size / TASK_SIZE;
        kernix_pool.tasks[i].end =
            (i + 1) * data_size / TASK_SIZE;
        kernix_pool.tasks[i].sum = 0.0;
    }

    kernix_pool.completed = 0;
    kernix_pool.generation++;

    pthread_cond_broadcast(&kernix_pool.work_cond);

    while (kernix_pool.completed < TASK_SIZE) {
        pthread_cond_wait(
            &kernix_pool.done_cond,
            &kernix_pool.mutex
        );
    }

    double total = 0.0;

    for (size_t i = 0; i < TASK_SIZE; i++) {
        total += kernix_pool.tasks[i].sum;
    }

    pthread_mutex_unlock(&kernix_pool.mutex);

    *result = total;
    return 0;
}

/* 기존 함수 이름을 유지해 사용 */
static inline int test_function(float_tensor* a)
{
    double total = 0.0;

    if (kernix_thread_pool_sum(a, &total) != 0) {
        fprintf(stderr, "텐서 합산 실패\n");
        return 1;
    }

    printf("합계: %.1f\n", total);
    return 0;
}

/* 프로그램 종료 시 한 번 호출 */
static void kernix_thread_pool_shutdown(void)
{
    if (!kernix_pool.initialized) {
        return;
    }

    pthread_mutex_lock(&kernix_pool.mutex);

    kernix_pool.stopping = 1;
    pthread_cond_broadcast(&kernix_pool.work_cond);

    pthread_mutex_unlock(&kernix_pool.mutex);

    for (size_t i = 0; i < TASK_SIZE; i++) {
        pthread_join(kernix_pool.threads[i], NULL);
    }

    pthread_cond_destroy(&kernix_pool.work_cond);
    pthread_cond_destroy(&kernix_pool.done_cond);
    pthread_mutex_destroy(&kernix_pool.mutex);

    kernix_pool.initialized = 0;
}


#define show_ints(tensor) show_ints_impl(tensor,#tensor)

#define show_floats(tensor) show_floats_impl(tensor,#tensor)

#define show_in8ts(tensor) show_in8ts_impl(tensor,#tensor)
#endif

