
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

int_tensor* int_zeros(int rows,int cols)
{
    int_tensor* p = malloc(sizeof(int_tensor));

    p->row = rows;
    p->col = cols;
    p->tensor = malloc(sizeof(*p->tensor)*rows*cols);

    memset(p->tensor,0,sizeof(int)*rows*cols);

    return p;

}

static inline void show_tensor_impl(int_tensor* tensor,const char *name) {

    printf("%s [\n", name);
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

#define show_tensor(tensor) show_tensor_impl(tensor,#tensor)

#endif

