#include <iostream>
#include <vpipes.hpp>
#include <string>
#include <queue>
int main() {

    using ids_type = size_t ;
    using val_type = string ;
    size_t result_size = 100 ;
    size_t input_size = 100 ;
    size_t batch_size = 10 ;
    auto result =  (ids_type *) malloc(sizeof(ids_type) * result_size);
    mondrian::vpipes::pipes::tid_materialize<val_type> ids_mat (result,&result_size);
    mondrian::vpipes::interval<ids_type> intr(0,input_size);
    auto predic =  mondrian::vpipes::predicates::batched_predicates<val_type>::unequal_to::micro_optimized_impl("M",true);
    auto predic_rm_A =  mondrian::vpipes::predicates::batched_predicates<val_type>::unequal_to::micro_optimized_impl("A",true);
    auto predic_rm_B =  mondrian::vpipes::predicates::batched_predicates<val_type>::unequal_to::micro_optimized_impl("B",true);
    queue<string> vals_to_fill ;
    vals_to_fill.push("A");
    vals_to_fill.push("B");
    vals_to_fill.push("C");
    mondrian::vpipes::block_copy<val_type,ids_type >::func_t  block_copy_func =[&vals_to_fill] (val_type *out, ids_type begin, ids_type end){

        auto distance =end-begin;
        for (auto i = 0 ; i<distance ;++i){
            *(out+i) = vals_to_fill.front();
            vals_to_fill.pop();
            vals_to_fill.push(*(out+i));
        }
    };

    mondrian::vpipes::pipes::filter<val_type> filter_A (&ids_mat,predic_rm_A,batch_size,true);
    mondrian::vpipes::pipes::filter<val_type> filter_B (&filter_A,predic_rm_B,batch_size,true);
    mondrian::vpipes::pipes::table_scan<val_type> tab (&filter_B, &intr , &intr+1 ,predic,block_copy_func,batch_size,batch_size,true);
    tab.start();
    std::cout<<"result size= " <<result_size<<std::endl;
    std::cout<<"###############printing the results###################"<<std::endl;
    for (auto itr=result; itr!= result+result_size ;++itr){
        std::cout<<"element:"<<*itr<<std::endl;
    }

    free(result);
    return 0;
}