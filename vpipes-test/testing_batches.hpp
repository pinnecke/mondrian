//
// Created by Mahmoud Mohsen on 3/15/17.
//

#pragma once

#include "gtest/gtest.h"
#include <vpipes.hpp>
#include <testing_utilities.hpp>

using namespace mondrian::vpipes;


TEST(Testbatches, TestAddStateAfterFilling) {
    auto batch_size = 15;
    mondrian::vpipes::batch<size_t> rat_batch(batch_size);
    auto ids = create_column(batch_size + 20, false);
    auto values = create_column(batch_size, false);
    auto indices_num = 20;
    auto indices = new size_t[indices_num];
    for (auto i = 0; i < indices_num; ++i) {
        indices[i] = i;
    }
    mondrian::vpipes::batch<size_t>::state rat_batch_state = mondrian::vpipes::batch<size_t>::state::non_full;
    rat_batch.add(&rat_batch_state, ids, values, indices, indices_num);
    EXPECT_NE(rat_batch_state, mondrian::vpipes::batch<size_t>::state::non_full)
                        << "after filling the batch the state will change";
    delete[] indices;
    delete_column(ids);
    delete_column(values);
    rat_batch.release();

}


TEST(Testbatches, TestAddStateAfterNotFilling) {
    auto batch_size = 15;
    mondrian::vpipes::batch<size_t> rat_batch(batch_size);
    auto ids = create_column(batch_size + 20, false);
    auto values = create_column(batch_size, false);
    auto indices_num = 5;
    auto indices = new size_t[indices_num];
    for (auto i = 0; i < indices_num; ++i) {
        indices[i] = i;
    }
    mondrian::vpipes::batch<size_t>::state rat_batch_state = mondrian::vpipes::batch<size_t>::state::non_full;
    rat_batch.add(&rat_batch_state, ids, values, indices, indices_num);
    EXPECT_EQ(rat_batch_state, mondrian::vpipes::batch<size_t>::state::non_full)
                        << "if the batch is not filled the state won't change";
    delete[] indices;
    delete_column(ids);
    delete_column(values);
    rat_batch.release();
}


TEST(Testbatches, TestAddReturnValue) {
    auto batch_size = 15;
    mondrian::vpipes::batch<size_t> rat_batch(batch_size);
    auto ids = create_column(batch_size + 20, false);
    auto values = create_column(batch_size, false);
    auto indices_num = 20;
    auto indices = new size_t[indices_num];
    for (auto i = 0; i < indices_num; ++i) {
        indices[i] = i;
    }
    mondrian::vpipes::batch<size_t>::state rat_batch_state = mondrian::vpipes::batch<size_t>::state::non_full;
    auto add_res = rat_batch.add(&rat_batch_state, ids, values, indices, indices_num);
    EXPECT_EQ(add_res, indices_num - batch_size)
                        << "add will return, how many elements it couldn't add because it is fulled";
    delete[] indices;
    delete_column(ids);
    delete_column(values);
    rat_batch.release();
}

TEST(TestBatches, TestReset) {
    auto batch_size = 15;
    mondrian::vpipes::batch<size_t> rat_batch(batch_size);
    auto ids = create_column(batch_size + 10, false);
    auto values = create_column(batch_size, false);
    auto indices_num = 20;
    auto indices = new size_t[indices_num];
    for (auto i = 0; i < indices_num; ++i) {
        indices[i] = i;
    }
    mondrian::vpipes::batch<size_t>::state rat_batch_state = mondrian::vpipes::batch<size_t>::state::non_full;
    auto indices_left = rat_batch.add(&rat_batch_state, ids, values, indices, indices_num);
    EXPECT_EQ(indices_left, indices_num - batch_size) << "this number denotes the left to be added";
    rat_batch.reset();
    indices_left = rat_batch.add(&rat_batch_state, ids, values, indices, indices_left);
    EXPECT_EQ(indices_left, 0) << "now all indices are added nothing left to be added";
    delete[] indices;
    delete_column(ids);
    delete_column(values);
    rat_batch.release();
}