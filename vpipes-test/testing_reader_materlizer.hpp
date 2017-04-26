//
// Created by Mahmoud Mohsen on 3/15/17.
//
#pragma once
#include "gtest/gtest.h"
#include <vpipes.hpp>
#include <testing_utilities.hpp>
#include <minimal_reader.hpp>
using namespace mondrian::vpipes;

mondrian::vpipes::point_null_copy::func_t null_copier = [] (mondrian::mtl::smart_bitmask *mask, const size_t *tupletids, size_t num_of_ids)
{
    mask->unset_all();
};

TEST(TestReading, TestBasicRead  ){
    size_t res_length = 500;
    auto batch_size = 10;
    auto input_length = 100;
    auto result = create_column(res_length, true);
    mondrian::vpipes::point_copy<size_t >::func_t ids_copier = [] (size_t *values, const size_t *tupletids, size_t num_of_ids)
    {
        for (auto i = 0; i< num_of_ids; ++i) {
            *(values+i) = *(tupletids+i);
        }
    };
    mondrian::vpipes::pipes::val_materialize<size_t> mat(result, &res_length);
    mondrian::vpipes::pipes::attribute_switch<size_t, size_t> proj(&mat, ids_copier, null_copier, batch_size);

    testing_vpipes_classes::minimal_reader<size_t > reader(&proj,mondrian::vpipes::predicates::batched_predicates<size_t >
    ::greater_equal::micro_optimized_impl(0,true),input_length,batch_size,batch_size);

    reader.read();

    //std::cout << "proj #batches [in]: " << proj.get_input_statistics()->num_batches << ", #empty " << proj.get_input_statistics()->num_empty_batches << std::endl;
    //std::cout << "proj #batches [out]: " << proj.get_output_statistics()->num_batches << ", #empty " << proj.get_input_statistics()->num_empty_batches << std::endl;
    //std::cout << "mat #batches  [in]: " << mat.get_input_statistics()->num_batches << ", #empty " << proj.get_input_statistics()->num_empty_batches << std::endl;

    auto input = reader.materlializer();

    EXPECT_EQ( has_same_vals(input,result,res_length) , true  );
    delete_column(result);
    delete_column(input);

}

TEST(TestReading, TestIfBatchSizeOddElementsNumEven  ){
    size_t res_length = 500;
    auto batch_size =9;
    auto  input_length= 100;
    auto  result = create_column(res_length, true);
    mondrian::vpipes::point_copy<size_t >::func_t ids_copier = [] (size_t *values, const size_t *tupletids, size_t num_of_ids)
    {
        for (auto i = 0; i< num_of_ids; ++i) {
            *(values+i) = *(tupletids+i);

        }
    };
    mondrian::vpipes::pipes::val_materialize<size_t> mat(result, &res_length);
    mondrian::vpipes::pipes::attribute_switch<size_t, size_t> proj(&mat, ids_copier, null_copier, batch_size);

    testing_vpipes_classes::minimal_reader<size_t > reader(&proj,mondrian::vpipes::predicates::batched_predicates<size_t >
    ::greater_equal::micro_optimized_impl(0,true),input_length,batch_size,batch_size);

    reader.read();

    auto input = reader.materlializer();
    EXPECT_EQ( has_same_vals(input,result,res_length) , true  );
    delete_column(result);
    delete_column(input);

}
//
TEST(TestReading, TestIfBatchSizeEvenElementsNumOdd  ){
    size_t res_length = 500;
    auto batch_size =10;
    auto  input_length= 93;
    auto  result = create_column(res_length, true);
    mondrian::vpipes::point_copy<size_t >::func_t ids_copier = [] (size_t *values, const size_t *tupletids, size_t num_of_ids)
    {
        for (auto i = 0; i< num_of_ids; ++i) {
            *(values+i) = *(tupletids+i);

        }
    };
    mondrian::vpipes::pipes::val_materialize<size_t> mat(result, &res_length);
    mondrian::vpipes::pipes::attribute_switch<size_t, size_t> proj(&mat, ids_copier, null_copier, batch_size);

    testing_vpipes_classes::minimal_reader<size_t > reader(&proj,mondrian::vpipes::predicates::batched_predicates<size_t >
    ::greater_equal::micro_optimized_impl(0,true),input_length,batch_size,batch_size);

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
////    auto batch_size =50;
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
////    toolkit::materialize<size_t > mat (result,&result_size,materialize1,batch_size);
////    auto read = toolkit::reader<std::size_t >(&mat,input ,input+num_elements ,batch_size);
////    read.start();
////    EXPECT_EQ( has_same_vals(input,result,result_size) , true  );
////    delete_column<size_t >(result);
////    delete_column<size_t >(input);
////
////}
//
