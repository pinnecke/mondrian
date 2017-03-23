//
// Created by Mahmoud Mohsen on 3/15/17.
//
#pragma once
#include "gtest/gtest.h"
#include <vpipes.hpp>
#include <testing_utilities.hpp>
#include <minimal_reader.hpp>
using namespace mondrian::vpipes;

TEST(TestReading, TestBasicRead  ){
    size_t res_length = 500;
    auto chunk_size =10;
    auto  input_length= 100;
    auto  result = create_column(res_length, true);
    mondrian::vpipes::point_copy<size_t >::func_t ids_copier = [] (size_t *out, const size_t *tupletids, size_t num_of_ids)
    {
        for (auto i = 0; i< num_of_ids; ++i) {
            *(out+i) = *(tupletids+i);

        }
    };
    mondrian::vpipes::pipes::materialize<size_t> mat(result,&res_length,ids_copier,chunk_size);

    testing_vpipes_classes::minimal_reader<size_t > reader(&mat,mondrian::vpipes::predicates::batched_predicates<size_t >
    ::greater_equal::micro_optimized_impl(0,true),input_length,chunk_size,chunk_size);

    reader.read();

    auto input = reader.materlializer();
    EXPECT_EQ( has_same_vals(input,result,res_length) , true  );
    delete_column(result);
    delete_column(input);

}

TEST(TestReading, TestIfChunkSizeOddElementsNumEven  ){
    size_t res_length = 500;
    auto chunk_size =9;
    auto  input_length= 100;
    auto  result = create_column(res_length, true);
    mondrian::vpipes::point_copy<size_t >::func_t ids_copier = [] (size_t *out, const size_t *tupletids, size_t num_of_ids)
    {
        for (auto i = 0; i< num_of_ids; ++i) {
            *(out+i) = *(tupletids+i);

        }
    };
    mondrian::vpipes::pipes::materialize<size_t> mat(result,&res_length,ids_copier,chunk_size);

    testing_vpipes_classes::minimal_reader<size_t > reader(&mat,mondrian::vpipes::predicates::batched_predicates<size_t >
    ::greater_equal::micro_optimized_impl(0,true),input_length,chunk_size,chunk_size);

    reader.read();

    auto input = reader.materlializer();
    EXPECT_EQ( has_same_vals(input,result,res_length) , true  );
    delete_column(result);
    delete_column(input);

}
//
TEST(TestReading, TestIfChunkSizeEvenElementsNumOdd  ){
    size_t res_length = 500;
    auto chunk_size =10;
    auto  input_length= 93;
    auto  result = create_column(res_length, true);
    mondrian::vpipes::point_copy<size_t >::func_t ids_copier = [] (size_t *out, const size_t *tupletids, size_t num_of_ids)
    {
        for (auto i = 0; i< num_of_ids; ++i) {
            *(out+i) = *(tupletids+i);

        }
    };
    mondrian::vpipes::pipes::materialize<size_t> mat(result,&res_length,ids_copier,chunk_size);

    testing_vpipes_classes::minimal_reader<size_t > reader(&mat,mondrian::vpipes::predicates::batched_predicates<size_t >
    ::greater_equal::micro_optimized_impl(0,true),input_length,chunk_size,chunk_size);

    reader.read();

    auto input = reader.materlializer();
    EXPECT_EQ( has_same_vals(input,result,res_length) , true  );
    delete_column(result);
    delete_column(input);

}
//
//
//// Testing failed, accessing none allocated memory
//
////TEST(TestReading, TestIfResultLessThanInput  ){
////    size_t num_elements =100;
////    auto input = create_column  <size_t> (num_elements,false,true);
////    auto chunk_size =50;
////    size_t result_size =10;
////    auto  result = create_column<size_t>(result_size);
////    mondrian::vpipes::functional::batched_materializes<size_t>::func_t materialize1 = [] (size_t *out_begin, size_t *out_end,
////                                                                                          const size_t *begin, const size_t*end)
////    {
////        assert (out_end - out_begin >= end - begin);
////        size_t distance = (end - begin);
////        for (size_t i = 0; i != distance; ++i) {
////            *(out_begin+i) = *(begin+i);
////        }
////    };
////    toolkit::materialize<size_t > mat (result,&result_size,materialize1,chunk_size);
////    auto read = toolkit::reader<std::size_t >(&mat,input ,input+num_elements ,chunk_size);
////    read.start();
////    EXPECT_EQ( has_same_vals(input,result,result_size) , true  );
////    delete_column<size_t >(result);
////    delete_column<size_t >(input);
////
////}
//
