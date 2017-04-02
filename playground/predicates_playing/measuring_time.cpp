//
// Created by Mahmoud Mohsen on 3/15/17.
//


#include <vpipes.hpp>
#include <chrono>
#include "measure_predicate_time.hpp"
#include <fstream>
#include "randoms_generator.hpp"
#include <iomanip>



#define PATH_TO_RESULTS_FILE "/home/pegasus/results/results_file6.csv"

int main() {

    using pred_type_branch_free = mondrian::vpipes::predicates::batched_predicates<size_t>::less_than:: branch_free_impl;
    using pred_type_micro_optimized = mondrian::vpipes::predicates::batched_predicates<size_t>::less_than::micro_optimized_impl;
    using pred_type_optimized_branch_free = mondrian::vpipes::predicates::batched_predicates<size_t>::less_than::optimized_branch_free_impl;
    size_t res_size =1000*1000*100*4;
    size_t input_size =1000*1000*100*4 ;
    std::size_t  batch_size =1000*1000*4;
    random_gen::higher_bound=input_size;

    random_gen::allocate_random_container(input_size);

    mondrian::vpipes::point_copy<size_t >::func_t point_copy_func = [] (size_t *out, const size_t *tupletids, size_t num_of_ids)
    {
        for (auto i = 0; i< num_of_ids; ++i) {
            *(out+i) = *(tupletids+i);

        }
    };

    mondrian::vpipes::block_copy<size_t>::func_t block_copy_func = [](size_t *out, size_t begin, size_t end){
        auto distance = end- begin;
        for (auto i = 0 ; i<distance ;++i){
            *(out+i) =random_gen::gen_random_number();
        }
    };
    random_gen::filled =false;

  /*
   *
   * the excution of the following block just to generate the random numbers,
   * and assigns them to the random container (not considered in calculating timing results)
   */
    std::cout<<"generating random values"<<std::endl;
    measure_pred_time measurer (res_size,input_size,batch_size,point_copy_func,block_copy_func);

        for (auto i =0 ; i < 11 ; i+=2 ){
            size_t pred_val = static_cast<size_t>  (random_gen::higher_bound *  (i/10.0 )    )  ;
            auto predicate=   pred_type_branch_free(pred_val,true);
            double time_res =measurer.measure_sampled_time<pred_type_branch_free>(1,predicate);
            double sf = (1.0*measurer.get_result_size()) /  (1.0 *input_size)  ;
            random_gen::filled=true;
            break;
        }

    std::cout<<"profiling branch free"<<std::endl;
    random_gen::reset =true;

    std::ofstream results_file (PATH_TO_RESULTS_FILE);
    if (results_file.is_open()){
        for (auto i =0 ; i < 11 ; i+=2 ){
            size_t pred_val = static_cast<size_t>  (random_gen::higher_bound *  (i/10.0 )    )  ;
            auto predicate=   pred_type_branch_free(pred_val,true);
            double time_res =measurer.measure_sampled_time<pred_type_branch_free>(10,predicate);
            double sf = (1.0*measurer.get_result_size()) /  (1.0 *input_size)  ;
            results_file<< std::fixed << std::setprecision(1) <<sf <<","<<time_res<<","<<"branch_free"<<"\n";
        }
    }else{
        std::cerr<<"could not open  the results file"<<endl;
    }
    std::cout<<"profiling mirco optimized"<<std::endl;
    random_gen::reset =true;
    if (results_file .is_open()){
        for (auto i =0 ; i < 11 ; i+=2 ){
            size_t pred_val = static_cast<size_t>  (random_gen::higher_bound *  (i/10.0 )    )  ;

            auto predicate=  pred_type_micro_optimized(pred_val,true);
            double time_res =measurer.measure_sampled_time<pred_type_micro_optimized>(10,predicate);
            double sf = (1.0*measurer.get_result_size()) /  (1.0 *input_size)  ;
            results_file<< std::fixed << std::setprecision(1) <<sf <<","<<time_res<<","<<"micro_optimizied"<<"\n";
        }
    }else{
        std::cerr<<"could not open the results file"<<endl;
    }
    std::cout<<"profiling optimized branch free"<<std::endl;
    random_gen::reset =true;

    if (results_file .is_open()){
        for (auto i =0 ; i < 11 ; i+=2 ){
            size_t pred_val = static_cast<size_t>  (random_gen::higher_bound *  (i/10.0 )    )  ;

            auto predicate=  pred_type_optimized_branch_free (pred_val,true);
            double time_res =measurer.measure_sampled_time<pred_type_optimized_branch_free>(10,predicate);
            double sf = (1.0*measurer.get_result_size()) /  (1.0 *input_size)  ;
            results_file<< std::fixed << std::setprecision(1) <<sf <<","<<time_res<<","<<"optimizied_branch_free"<<"\n";
        }
    }else{
        std::cerr<<"could not open the results file"<<endl;
    }

    results_file.close();
    random_gen::delete_random_container();


    return EXIT_SUCCESS;
}