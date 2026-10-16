#include "kernix.h"
#include <stdio.h>

int main(){
    
    int_tensor* train_x = int_zeros(2,2);
    int_tensor* train_y = int_zeros(2,2);

    train_x->tensor[0] = 6;
    train_x->tensor[1] = 3;
    train_x->tensor[2] = 7;
    train_x->tensor[3] = 9;
    
    train_y->tensor[0] = 2;
    train_y->tensor[1] = 9;
    train_y->tensor[2] = 5;
    train_y->tensor[3] = 1;

    show_ints(train_x);
    show_ints(train_y);

    int_tensor* c = int_matmul(train_x,train_y);
    show_ints(c);

    int_free(train_x);
    int_free(train_y);

    return 0;
}