#include "kernix.h"

int main(){
    
    float_tensor* train_x = float_zeros(256,256);
    float_tensor* train_y = float_zeros(256,256);

    train_x->tensor[0] = 6.4;
    train_x->tensor[1] = 3.2;
    train_x->tensor[2] = 7.1;
    train_x->tensor[3] = 9.2;
    
    train_y->tensor[0] = 2.3;
    train_y->tensor[1] = 9.8;
    train_y->tensor[2] = 5.7;
    train_y->tensor[3] = 1.4;

    show_floats(train_x);
    show_floats(train_y);

    //float_tensor* c = float_matmul(train_x,train_y);
    //float_tensor* d = floats_add(train_x,train_y);
    printf("%0.2f\n",float_dot(train_x,train_y));
    //show_floats(c);
    //show_floats(d);

    float_free(train_x);
    float_free(train_y);
    //float_free(c);
    //float_free(d);

    return 0;
}