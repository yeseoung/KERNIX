#include <kernix.h>

int main(){
    
    int_tensor* train_x = int_zeros(4,4);
    show_tensor(train_x);

    free(train_x->tensor);
    return 0;
}