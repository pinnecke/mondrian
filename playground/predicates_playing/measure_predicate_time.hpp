//
// Created by Mahmoud Mohsen on 3/29/17.
//
#pragma once

#include <vpipes.hpp>
#include <chrono>
#include <profiling.hpp>



class measure_pred_time{

   using point_copy_type = mondrian::vpipes::point_copy<size_t >::func_t;
    using block_copy_type = mondrian::vpipes::block_copy<size_t>::func_t;
    private:
        size_t m_res_size ;
        size_t m_input_size;
        size_t m_batch_size;
        point_copy_type  m_point_copy_func ;
        block_copy_type  m_block_copy_func;
        size_t  loop_res_size ;
    public:
        measure_pred_time (   size_t p_res_size, size_t p_input_size,size_t p_batch_size,
                              point_copy_type  p_point_copy_func,
            block_copy_type  p_block_copy_func):m_res_size(p_res_size),m_input_size(p_input_size),
                                               m_batch_size(p_batch_size),
                                               m_point_copy_func(p_point_copy_func),m_block_copy_func(p_block_copy_func)
        {
                loop_res_size = 0;
        }

        template <class predicate_type>
        double measure_sampled_time (int num_samples ,predicate_type p_predicate){
            size_t  current_duration = 0;
            loop_res_size =m_res_size;
            mondrian::vpipes::interval<size_t> all_tuplet_ids(0, m_input_size);
            auto index = 0 ;
            auto loc_pred = p_predicate;
            auto loc_block_copy =m_block_copy_func;
            auto loc_batch_size =m_batch_size;
            while (num_samples -index){
                auto result = (std::size_t *) malloc (m_res_size * sizeof(size_t));
                loop_res_size =0;
                mondrian::vpipes::pipes::materialize<size_t> mat(result,&loop_res_size,m_point_copy_func,m_batch_size);
                current_duration += mondrian::utils::profiling::measure<std::chrono::nanoseconds>::execute(
                        [&mat, &all_tuplet_ids ,&loc_pred, &loc_block_copy,&loc_batch_size]() {
                            auto loc_table = mondrian::vpipes::  pipes::table_scan<size_t >(&mat, &all_tuplet_ids, &all_tuplet_ids + 1,loc_pred, loc_block_copy ,loc_batch_size, loc_batch_size);
                            loc_table.start();
                        });
                free(result);
                ++index;
            }
            double current_duration_d =( current_duration / double(num_samples) ) / 1000000000.0f;
            return current_duration_d;
        }
        size_t  get_result_size(){
            return loop_res_size;
        }

};