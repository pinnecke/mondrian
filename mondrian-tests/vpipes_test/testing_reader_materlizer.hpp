//
// Created by Mahmoud Mohsen on 3/15/17.
//
#pragma once
#include "gtest/gtest.h"
#include <vpipes.hpp>
#include <testing_utilities.hpp>
using namespace mondrian::vpipes;

TEST(TestReading, TestBasicRead  ){
    size_t num_elements =10;
    auto input = create_column  <size_t> (num_elements,false,true);
    auto chunk_size =10;
    auto  result = create_column<size_t>(num_elements);
    mondrian::vpipes::functional::batched_materializes<size_t>::func_t materialize1 = [] (size_t *out_begin, size_t *out_end,
                                                                                          const size_t *begin, const size_t*end)
    {
        assert (out_end - out_begin >= end - begin);
        size_t distance = (end - begin);
        for (size_t i = 0; i != distance; ++i) {
            *(out_begin+i) = *(begin+i);
        }
    };
    toolkit::materialize<size_t > mat (result,&num_elements,materialize1,chunk_size);
    auto read = toolkit::reader<std::size_t >(&mat,input ,input+num_elements ,chunk_size);
    read.start();
    EXPECT_EQ( has_same_vals(input,result,num_elements) , true  );
    delete_column<size_t >(result);
    delete_column<size_t >(input);

}

TEST(TestReading, TestIfChunkSizeOddElementsNumEven  ){
    size_t num_elements =100;
    auto input = create_column  <size_t> (num_elements,false,true);
    auto chunk_size =3;
    auto  result = create_column<size_t>(num_elements);
    mondrian::vpipes::functional::batched_materializes<size_t>::func_t materialize1 = [] (size_t *out_begin, size_t *out_end,
                                                                                          const size_t *begin, const size_t*end)
    {
        assert (out_end - out_begin >= end - begin);
        size_t distance = (end - begin);
        for (size_t i = 0; i != distance; ++i) {
            *(out_begin+i) = *(begin+i);
        }
    };
    toolkit::materialize<size_t > mat (result,&num_elements,materialize1,chunk_size);
    auto read = toolkit::reader<std::size_t >(&mat,input ,input+num_elements ,chunk_size);
    read.start();
    EXPECT_EQ( has_same_vals(input,result,num_elements) , true  );
    delete_column<size_t >(result);
    delete_column<size_t >(input);

}

TEST(TestReading, TestIfChunkSizeEvenElementsNumOdd  ){
    size_t num_elements =93;
    auto input = create_column  <size_t> (num_elements,false,true);
    auto chunk_size =4;
    auto  result = create_column<size_t>(num_elements);
    mondrian::vpipes::functional::batched_materializes<size_t>::func_t materialize1 = [] (size_t *out_begin, size_t *out_end,
                                                                                          const size_t *begin, const size_t*end)
    {
        assert (out_end - out_begin >= end - begin);
        size_t distance = (end - begin);
        for (size_t i = 0; i != distance; ++i) {
            *(out_begin+i) = *(begin+i);
        }
    };
    toolkit::materialize<size_t > mat (result,&num_elements,materialize1,chunk_size);
    auto read = toolkit::reader<std::size_t >(&mat,input ,input+num_elements ,chunk_size);
    read.start();
    EXPECT_EQ( has_same_vals(input,result,num_elements) , true  );
    delete_column<size_t >(result);
    delete_column<size_t >(input);

}


// Testing failed, accessing none allocated memory

//TEST(TestReading, TestIfResultLessThanInput  ){
//    size_t num_elements =100;
//    auto input = create_column  <size_t> (num_elements,false,true);
//    auto chunk_size =50;
//    size_t result_size =10;
//    auto  result = create_column<size_t>(result_size);
//    mondrian::vpipes::functional::batched_materializes<size_t>::func_t materialize1 = [] (size_t *out_begin, size_t *out_end,
//                                                                                          const size_t *begin, const size_t*end)
//    {
//        assert (out_end - out_begin >= end - begin);
//        size_t distance = (end - begin);
//        for (size_t i = 0; i != distance; ++i) {
//            *(out_begin+i) = *(begin+i);
//        }
//    };
//    toolkit::materialize<size_t > mat (result,&result_size,materialize1,chunk_size);
//    auto read = toolkit::reader<std::size_t >(&mat,input ,input+num_elements ,chunk_size);
//    read.start();
//    EXPECT_EQ( has_same_vals(input,result,result_size) , true  );
//    delete_column<size_t >(result);
//    delete_column<size_t >(input);
//
//}

