# KERNIX

**KERNIX** is a lightweight C library project focused on matrix operations, low-level memory management, and performance-oriented AI system development.

The project aims to build a simple and efficient foundation for machine learning systems using C, with an emphasis on memory control, computational efficiency, and minimal dependencies.

### Goals
- Simple and intuitive matrix APIs
- Efficient memory management
- Fundamental matrix and tensor operations
- Low-level performance optimization
- A foundation for building machine learning systems in C

### Current Features
- Matrix visualization through `show_matrix()`
- Header-only implementation for easy integration

### Example

```c
#include <kernix.h>
#include <stdlib.h>

int main(void) {
    int rows = 3;
    int cols = 3;

    int *train_x = calloc(rows * cols, sizeof(int));
    if (train_x == NULL) {
        return 1;
    }

    show_matrix(train_x, rows, cols);

    free(train_x);
    return 0;
}
```

### Vision

KERNIX aims to explore how machine learning infrastructure can be built from the ground up in C, providing greater control over memory, computation, and system-level optimization.
