#include <kernix.h>

int main(){
    int_tensor* p = int_zeros(10,10);
    show_tensor(p);
    
    free(p->tensor);
    return 0;
}