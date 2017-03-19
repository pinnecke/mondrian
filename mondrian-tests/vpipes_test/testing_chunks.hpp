//
// Created by Mahmoud Mohsen on 3/15/17.
//

#pragma once

#include "gtest/gtest.h"
#include <vpipes.hpp>
#include <testing_utilities.hpp>

using namespace mondrian::vpipes;


TEST(Testchunks, TestingSimpleAdd) {
    auto chunk_size = 5;
    auto expected_max_cursor = 0;
    chunk<size_t> rat_chunk(chunk_size);
    chunk<size_t>::state chunk_state;
    for (auto i = 0; i < chunk_size + 10; ++i) {
        chunk_state = rat_chunk.add(i);
        if (chunk_state == chunk<size_t>::state::full) {
            expected_max_cursor = i + 1;
            break;
        }
    }

    EXPECT_EQ(expected_max_cursor, chunk_size);
    rat_chunk.release();
}

TEST(Testchunks, TestbatchAddState) {
    auto chunk_size = 5;
    chunk<size_t> rat_chunk(chunk_size);
    size_t *donator = create_column<size_t>(15, false);
    chunk<size_t>::state chunk_state = chunk<size_t>::state::non_full;
    rat_chunk.add(&chunk_state, donator, donator + 15);
    EXPECT_EQ(chunk_state, chunk<size_t>::state::full);
    delete_column<size_t>(donator);
    rat_chunk.release();
}


TEST(Testchunks, TestbatchAddValue) {
    auto chunk_size = 7;
    auto expected_max_cursor = 0;
    chunk<size_t> rat_chunk(chunk_size);
    size_t *donator = create_column<size_t>(15, false);
    chunk<size_t>::state chunk_state = chunk<size_t>::state::non_full;
    expected_max_cursor = *rat_chunk.add(&chunk_state, donator, donator + 15);
    EXPECT_EQ(expected_max_cursor, chunk_size);
    delete_column<size_t>(donator);
    rat_chunk.release();
}

TEST(TestChunks, TestReset) {
    auto chunk_size = 5;
    auto total_insertion_times = chunk_size + 10;
    auto expected_max_cursor = 0;
    chunk<size_t> rat_chunk(chunk_size);
    chunk<size_t>::state chunk_state;
    for (expected_max_cursor = 0; expected_max_cursor < total_insertion_times; ++expected_max_cursor) {
        chunk_state = rat_chunk.add(expected_max_cursor);
        if (chunk_state == chunk<size_t>::state::full) {
            rat_chunk.reset();
        }
    }
    EXPECT_EQ(expected_max_cursor, total_insertion_times);
    rat_chunk.release();
}