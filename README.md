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
#include "kernix.h"

int main(){
    
    int_tensor* train_x = int_zeros(2,2);

    show_ints(train_x);

    int_free(train_x);

    return 0;
}
```

### Vision

KERNIX aims to explore how machine learning infrastructure can be built from the ground up in C, providing greater control over memory, computation, and system-level optimization.
