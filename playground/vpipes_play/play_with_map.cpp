#include <iostream>
#include <vpipes.hpp>

int main() {

    using ids_type = size_t ;
    using input_val_type = size_t ;
    using output_val_type = bool ;
    size_t result_size = 100 ;
    size_t input_size = 100 ;
    size_t batch_size = 10 ;
    auto result =  (output_val_type*) malloc(sizeof(output_val_type) * result_size);
    mondrian::vpipes::pipes::val_materialize<output_val_type> val_mat (result,&result_size);
    mondrian::vpipes::interval<ids_type> intr(0,input_size);
    auto predic =  mondrian::vpipes::predicates::batched_predicates<input_val_type>::unequal_to::micro_optimized_impl(-1,true);

    mondrian::vpipes::block_copy<input_val_type ,ids_type >::func_t  block_copy_func =[] (input_val_type *out, ids_type begin, ids_type end){

        auto distance =end-begin;
        for (auto i = 0 ; i<distance ;++i){
            *(out+i) =begin+i;
        }
    };
    auto map_func = mondrian::vpipes::maps::indicators<input_val_type>::greater_than::straightforward_impl(50);

    mondrian::vpipes::pipes::map<input_val_type,output_val_type> map_consmr (&val_mat,map_func,batch_size);
    mondrian::vpipes::pipes::table_scan<input_val_type> tab (&map_consmr, &intr , &intr+1 ,predic,block_copy_func,batch_size,batch_size,true);
    tab.start();
    std::cout<<"result size= " <<result_size<<std::endl;
    std::cout<<"###############printing the results###################"<<std::endl;
    for (auto itr=result; itr!= result+result_size ;++itr){
        std::cout<<"element:"<<*itr<<std::endl;
    }

    free(result);
    return 0;
}