
#ifndef KERNIX_H
#define KERNIX_H

#include <stdio.h>

static inline void kx_matrix_show(
    int *matrix,
    int rows,
    int cols,
    const char *name
) {
    printf("%s [\n", name);

    for (int i = 0; i < rows; i++) {
        printf("  { ");

        for (int j = 0; j < cols; j++) {
            printf("%3d", matrix[i * cols + j]);

            if (j < cols - 1) {
                printf(" , ");
            }
        }

        printf(" }\n");
    }

    printf("]\n");
}

#define show_matrix(matrix, rows, cols) \
    kx_matrix_show(matrix, rows, cols, #matrix)

#endif

