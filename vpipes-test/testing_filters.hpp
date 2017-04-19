//
// Created by Mahmoud Mohsen on 3/16/17.
//

#pragma once

#include "gtest/gtest.h"
#include <vpipes.hpp>
#include <testing_utilities.hpp>

using namespace mondrian::vpipes;

TEST(TESTfilters, BasicFiltersTest) {
    size_t res_length = 500;
    auto batch_size =20;
    auto  input_length= 50;
    auto  result = create_column(res_length, true);
    auto predicate_value = 5 ;

    mondrian::vpipes::point_copy<size_t >::func_t ids_copier = [] (size_t *values, const size_t *tupletids, size_t num_of_ids)
    {
        for (auto i = 0; i< num_of_ids; ++i) {
            *(values+i) = *(tupletids+i);

        }
    };
    mondrian::vpipes::pipes::val_materialize<size_t> mat(result, &res_length);
    mondrian::vpipes::pipes::project<size_t, size_t> proj(&mat, ids_copier, null_copier, batch_size);

    mondrian::vpipes::pipes::filter<size_t> filter_nums_more_5(&proj , mondrian::vpipes::predicates::batched_predicates<size_t >
    ::greater_equal::micro_optimized_impl(predicate_value,true),batch_size, true);

    testing_vpipes_classes::minimal_reader<size_t > reader(&filter_nums_more_5,mondrian::vpipes::predicates::batched_predicates<size_t >
    ::greater_equal::micro_optimized_impl(0,true),input_length,batch_size,batch_size);

    reader.read();

    auto expected_result = generate_expected_result(input_length,[predicate_value] (size_t val) -> size_t {
        return ! (val >=predicate_value);
    });

    EXPECT_EQ( has_same_vals(expected_result,result,res_length) , true  );
    delete_column(result);
    delete_column(expected_result);

}


TEST(TESTfilters, CascadingFilters) {
    size_t res_length = 500;
    auto batch_size =20;
    auto  input_length= 50;
    auto  result = create_column(res_length, true);
    auto lower_bound = 5 ;
    auto upper_bound = 30 ;
    mondrian::vpipes::point_copy<size_t >::func_t ids_copier = [] (size_t *values, const size_t *tupletids, size_t num_of_ids)
    {
        for (auto i = 0; i< num_of_ids; ++i) {
            *(values+i) = *(tupletids+i);

        }
    };
    mondrian::vpipes::pipes::val_materialize<size_t> mat(result, &res_length);
    mondrian::vpipes::pipes::project<size_t, size_t> proj(&mat, ids_copier, null_copier, batch_size);

    mondrian::vpipes::pipes::filter<size_t> filter_nums_more_5(&proj, mondrian::vpipes::predicates::batched_predicates<size_t >
    ::greater_equal::micro_optimized_impl(5,true),batch_size, true);

    mondrian::vpipes::pipes::filter<size_t> filter_nums_less_20(&filter_nums_more_5 , mondrian::vpipes::predicates::batched_predicates<size_t >
    ::less_equal::micro_optimized_impl(30,true),batch_size, true);


    testing_vpipes_classes::minimal_reader<size_t > reader(&filter_nums_less_20,mondrian::vpipes::predicates::batched_predicates<size_t >
    ::greater_equal::micro_optimized_impl(0,true),input_length,batch_size,batch_size);

    reader.read();

    auto expected_result = generate_expected_result(input_length,[lower_bound,upper_bound] (size_t val) -> size_t {
        return ! (val >=lower_bound && val<=upper_bound  );
    });

    EXPECT_EQ( has_same_vals(expected_result,result,res_length) , true  );
    delete_column(result);
    delete_column(expected_result);

}


TEST(TESTfilters, NoConditionSatisfied) {
    size_t res_length = 50;
    auto batch_size =20;
    auto  input_length= 50;
    auto  result = create_column(res_length, true);
    auto expected_result =create_column(res_length, true);
    auto predicate_value = 100 ;
    mondrian::vpipes::point_copy<size_t >::func_t ids_copier = [] (size_t *values, const size_t *tupletids, size_t num_of_ids)
    {
        for (auto i = 0; i< num_of_ids; ++i) {
            *(values+i) = *(tupletids+i);

        }
    };
    mondrian::vpipes::pipes::val_materialize<size_t> mat(result, &res_length);
    mondrian::vpipes::pipes::project<size_t, size_t> proj(&mat, ids_copier, null_copier, batch_size);

    mondrian::vpipes::pipes::filter<size_t> filter_nums_more_100(&proj , mondrian::vpipes::predicates::batched_predicates<size_t >
    ::greater_equal::micro_optimized_impl(predicate_value,true),batch_size, true);

    testing_vpipes_classes::minimal_reader<size_t > reader(&filter_nums_more_100,mondrian::vpipes::predicates::batched_predicates<size_t >
    ::greater_equal::micro_optimized_impl(0,true),input_length,batch_size,batch_size);

    reader.read();

    EXPECT_EQ( has_same_vals(expected_result,result,res_length) , true  );
    delete_column(result);
    delete_column(expected_result);

}