#include <iostream>
#include <vpipes.hpp>

int main() {

    using ids_type = size_t ;
    using val_type = size_t ;
    size_t result_size = 100 ;
    size_t input_size = 100 ;
    size_t batch_size = 10 ;
    auto result =  (val_type*) malloc(sizeof(val_type) * result_size);
    mondrian::vpipes::pipes::val_materialize<val_type> val_mat (result,&result_size);
    mondrian::vpipes::interval<ids_type> intr(0,input_size);
    auto predic =  mondrian::vpipes::predicates::batched_predicates<val_type>::greater_than::micro_optimized_impl(5,true);

    mondrian::vpipes::block_copy<val_type,ids_type >::func_t  block_copy_func =[] (val_type *out, ids_type begin, ids_type end){

        auto distance =end-begin;
        for (auto i = 0 ; i<distance ;++i){
                *(out+i) =begin+i ;
        }
    };

    mondrian::vpipes::pipes::table_scan<val_type> tab (&val_mat, &intr , &intr+1 ,predic,block_copy_func,batch_size,batch_size,true);
    tab.start();
    std::cout<<"result size= " <<result_size<<std::endl;
    std::cout<<"###############printing the results###################"<<std::endl;
    for (auto itr=result; itr!= result+result_size ;++itr){
        std::cout<<"element:"<<*itr<<std::endl;
    }

    free(result);
    return 0;
}