
#ifndef KERNIX_H
#define KERNIX_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

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

int_tensor* int_matmul(const int_tensor* a,const int_tensor* b){
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

#define show_ints(tensor) show_ints_impl(tensor,#tensor)

#define show_floats(tensor) show_floats_impl(tensor,#tensor)

#define show_in8ts(tensor) show_in8ts_impl(tensor,#tensor)
#endif

