//
// Created by Mahmoud Mohsen on 3/17/17.
//
#pragma once
#include "gtest/gtest.h"
#include <vpipes.hpp>
#include <testing_utilities.hpp>


TEST(TestTableScan,TestBasicFunctionality){
    size_t res_length = 50;
    auto batch_size =20;
    auto  input_length= 50;
    auto  result = create_column(res_length, true);
    auto predicate_value = 5 ;

    mondrian::vpipes::point_copy<size_t >::func_t ids_copier = [] (size_t *out, const size_t *tupletids, size_t num_of_ids)
    {
        for (auto i = 0; i< num_of_ids; ++i) {
            *(out+i) = *(tupletids+i);

        }
    };
    mondrian::vpipes::pipes::val_materialize<size_t> mat(result, &res_length);
    mondrian::vpipes::pipes::project<size_t, size_t> proj(&mat, ids_copier, batch_size);


    interval<size_t> all_tuplet_ids(0, input_length);


    mondrian::vpipes::block_copy<size_t>::func_t loc_block_copy = [](size_t *out, size_t begin, size_t end){
        auto distance = end- begin;
        for (auto i = 0 ; i<distance ;++i){
            *(out+i) =begin+i;
        }
    };


    auto loc_table = pipes::table_scan<size_t >(&proj, &all_tuplet_ids, &all_tuplet_ids + 1,mondrian::vpipes::predicates::batched_predicates<size_t >
                                                ::greater_equal::micro_optimized_impl(predicate_value,true),
                                                loc_block_copy   , batch_size, batch_size, true);

    loc_table.start();

    auto expected_result = generate_expected_result(input_length,[predicate_value] (size_t val) -> size_t {
        return ! (val >=predicate_value);
    });

    EXPECT_EQ( has_same_vals(expected_result,result,res_length) , true  );

    delete_column(result);
    delete_column(expected_result);

}

