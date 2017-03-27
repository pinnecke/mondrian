//
// Created by Mahmoud Mohsen on 3/15/17.
//

#pragma once

#include "gtest/gtest.h"
#include <vpipes.hpp>
#include <testing_utilities.hpp>

using namespace mondrian::vpipes;


TEST(Testchunks, TestAddStateAfterFilling) {
    auto chunk_size = 15 ;
    mondrian::vpipes::chunk<size_t > rat_chunk (chunk_size);
    auto ids = create_column(chunk_size+20,false);
    auto values = create_column(chunk_size,false);
    auto indices_num = 20 ;
    auto indices = new size_t [indices_num];
    for (auto i =0 ; i < indices_num; ++i) {
        indices [i] = i;
    }
    mondrian::vpipes::chunk<size_t >::state rat_chunk_state = mondrian::vpipes::chunk<size_t >::state::non_full;
    rat_chunk.add(&rat_chunk_state,ids,values,indices,indices_num);
    EXPECT_NE(rat_chunk_state,mondrian::vpipes::chunk<size_t >::state::non_full )<<"after filling the chunk the state will change";
    delete [] indices;
    delete_column(ids);
    delete_column(values);
    rat_chunk.release();

}


TEST(Testchunks, TestAddStateAfterNotFilling) {
    auto chunk_size = 15 ;
    mondrian::vpipes::chunk<size_t > rat_chunk (chunk_size);
    auto ids =create_column(chunk_size+20,false);
    auto values =create_column(chunk_size,false);
    auto indices_num = 5 ;
    auto indices = new size_t [indices_num];
    for (auto i =0 ; i <indices_num ;++i){
        indices [i] = i;
    }
    mondrian::vpipes::chunk<size_t >::state  rat_chunk_state =  mondrian::vpipes::chunk<size_t >::state::non_full;
    rat_chunk.add(&rat_chunk_state,ids,values,indices,indices_num);
    EXPECT_EQ(rat_chunk_state,mondrian::vpipes::chunk<size_t >::state::non_full )<<"if the chunk is not filled the state won't change";
    delete [] indices;
    delete_column(ids);
    delete_column(values);
    rat_chunk.release();

}


TEST(Testchunks, TestAddReturnValue) {
    auto chunk_size = 15 ;
    mondrian::vpipes::chunk<size_t > rat_chunk (chunk_size);
    auto ids =create_column(chunk_size+20,false);
    auto values =create_column(chunk_size,false);
    auto indices_num = 20 ;
    auto indices = new size_t [indices_num];
    for (auto i =0 ; i <indices_num ;++i){
        indices [i] = i;
    }
    mondrian::vpipes::chunk<size_t >::state  rat_chunk_state =  mondrian::vpipes::chunk<size_t >::state::non_full;
    auto add_res = rat_chunk.add(&rat_chunk_state,ids,values,indices,indices_num);
    EXPECT_EQ(add_res, indices_num - chunk_size )<<"add will return, how many elements it couldn't add because it is fulled";
    delete [] indices;
    delete_column(ids);
    delete_column(values);
    rat_chunk.release();

}

TEST(TestChunks, TestReset) {
    auto chunk_size = 15 ;
    mondrian::vpipes::chunk<size_t > rat_chunk (chunk_size);
    auto ids =create_column(chunk_size+10,false);
    auto values =create_column(chunk_size,false);
    auto indices_num = 20 ;
    auto indices = new size_t [indices_num];
    for (auto i =0 ; i <indices_num ;++i){
        indices [i] = i;
    }
    mondrian::vpipes::chunk<size_t >::state  rat_chunk_state =  mondrian::vpipes::chunk<size_t >::state::non_full;
    auto indices_left  = rat_chunk.add(&rat_chunk_state,ids,values,indices,indices_num);
    EXPECT_EQ(indices_left,indices_num-chunk_size )<<"this number denotes the left to be added";
    rat_chunk.reset();
    indices_left = rat_chunk.add(&rat_chunk_state,ids,values,indices,indices_left);
    EXPECT_EQ(indices_left,0 )<<"now all indices are added nothing left to be added";

    delete [] indices;
    delete_column(ids);
    delete_column(values);
    rat_chunk.release();

}