//
// Created by Mahmoud Mohsen on 4/6/17.
//

#pragma once

#include "gtest/gtest.h"
#include <vpipes.hpp>
#include <testing_utilities.hpp>
#include <minimal_reader.hpp>
#include <queue>

using namespace mondrian::vpipes::datastructures;

TEST(TestingDedupFirstFlavour, TestDedupForValues) {

    size_t res_length = 1500;
    auto batch_size = 10;
    auto input_length = 1500;
    auto result = (size_t *) malloc(res_length * sizeof(size_t));

    std::queue<size_t> repeated_values;
    repeated_values.push(1000);
    repeated_values.push(20);
    repeated_values.push(30);
    mondrian::vpipes::block_copy<size_t>::func_t loc_block_copy = [&repeated_values](size_t *out, size_t begin,
                                                                                     size_t end) {
        auto distance = end - begin;
        for (auto i = 0; i < distance; ++i) {
            *(out + i) = repeated_values.front();
            repeated_values.pop();
            repeated_values.push(*(out + i));
        }
    };

    mondrian::vpipes::pipes::val_materialize<size_t> mat(result, &res_length);
    red_black_tree<std::size_t> ids;
    red_black_tree<size_t> vals;
    mondrian::vpipes::pipes::dedup_1_flavour<size_t>::policy policy_used = mondrian::vpipes::pipes::dedup_1_flavour<size_t>::policy::values;
    mondrian::vpipes::pipes::dedup_1_flavour<size_t> ded(&mat, batch_size, &ids, &vals, policy_used, true);
    testing_vpipes_classes::minimal_reader<size_t> reader(&ded, mondrian::vpipes::predicates::batched_predicates<size_t>
    ::greater_than::micro_optimized_impl(0, true), loc_block_copy, input_length, batch_size, batch_size);

    reader.read();
    bool only_non_repeated = (res_length == repeated_values.size());

    for (auto itr = result; itr != result + res_length; ++itr) {
        if (*itr != repeated_values.front()) {
            only_non_repeated = false;
            break;
        }
        repeated_values.pop();
    }
    EXPECT_EQ(true, only_non_repeated);
    free(result);
}


TEST(TestingDedupFirstFlavour, TestDedupForIds) {

    size_t res_length = 1500;
    auto batch_size = 10;
    auto input_length = 1500;
    auto result = (size_t *) malloc(res_length * sizeof(size_t));
    std::queue<size_t> repeated_values;
    repeated_values.push(1000);
    repeated_values.push(20);
    repeated_values.push(30);
    mondrian::vpipes::block_copy<size_t>::func_t loc_block_copy = [&repeated_values](size_t *out, size_t begin,
                                                                                     size_t end) {
        auto distance = end - begin;
        for (auto i = 0; i < distance; ++i) {
            *(out + i) = repeated_values.front();
            repeated_values.pop();
            repeated_values.push(*(out + i));
        }
    };

    mondrian::vpipes::pipes::tid_materialize<size_t> mat(result, &res_length);
    red_black_tree<std::size_t> ids;
    red_black_tree<size_t> vals;
    mondrian::vpipes::pipes::dedup_1_flavour<size_t>::policy policy_used = mondrian::vpipes::pipes::dedup_1_flavour<size_t>::policy::Ids;
    mondrian::vpipes::pipes::dedup_1_flavour<size_t> ded(&mat, batch_size, &ids, &vals, policy_used, true);
    testing_vpipes_classes::minimal_reader<size_t> reader(&ded, mondrian::vpipes::predicates::batched_predicates<size_t>
    ::greater_than::micro_optimized_impl(0, true), loc_block_copy, input_length, batch_size, batch_size);
    reader.read();

    bool only_non_repeated = (res_length == 1);
    for (auto itr = result; itr != result + res_length; ++itr) {
        if (*itr != 666) {
            only_non_repeated = false;
            break;
        }
    }

    EXPECT_EQ(true, only_non_repeated);

    free(result);
}

TEST(TestingDedupFirstFlavour, TestDedupForboth) {

    size_t res_length = 5;
    auto batch_size = 2;
    auto input_length = 5;
    auto result = (size_t *) malloc(res_length * sizeof(size_t));
    std::queue<size_t> repeated_values;
    repeated_values.push(1000);
    repeated_values.push(20);
    repeated_values.push(30);
    repeated_values.push(15);
    repeated_values.push(1000);
    mondrian::vpipes::block_copy<size_t>::func_t loc_block_copy = [&repeated_values](size_t *out, size_t begin,
                                                                                     size_t end) {
        auto distance = end - begin;
        for (auto i = 0; i < distance; ++i) {
            *(out + i) = repeated_values.front();
            repeated_values.pop();
            repeated_values.push(*(out + i));
        }
    };

    mondrian::vpipes::pipes::val_materialize<size_t> mat(result, &res_length);
    red_black_tree<std::size_t> ids;
    red_black_tree<size_t> vals;
    mondrian::vpipes::pipes::dedup_1_flavour<size_t>::policy policy_used = mondrian::vpipes::pipes::dedup_1_flavour<size_t>::policy::both;
    mondrian::vpipes::pipes::dedup_1_flavour<size_t> ded(&mat, batch_size, &ids, &vals, policy_used, true);
    testing_vpipes_classes::minimal_reader<size_t> reader(&ded, mondrian::vpipes::predicates::batched_predicates<size_t>
    ::greater_than::micro_optimized_impl(0, true), loc_block_copy, input_length, batch_size, batch_size);

    reader.read();
    bool only_non_repeated = (res_length == (repeated_values.size() - 1));

    for (auto itr = result; itr != result + res_length; ++itr) {
        if (*itr != repeated_values.front()) {
            only_non_repeated = false;
            break;
        }
        repeated_values.pop();
    }

    EXPECT_EQ(true, only_non_repeated);
    free(result);
}

TEST(TestingDedupFirstFlavour, TestDedupIDsRepeatednoValueRepeated) {

    size_t res_length = 500;
    auto batch_size = 10;
    auto input_length = 500;
    auto original_res_length = res_length;
    auto result = (size_t *) malloc(res_length * sizeof(size_t));

    mondrian::vpipes::block_copy<size_t>::func_t loc_block_copy = [](size_t *out, size_t begin, size_t end) {
        auto distance = end - begin;
        for (auto i = 0; i < distance; ++i) {
            *(out + i) = begin + i;
        }
    };

    mondrian::vpipes::pipes::val_materialize<size_t> mat(result, &res_length);
    red_black_tree<std::size_t> ids;
    red_black_tree<size_t> vals;
    mondrian::vpipes::pipes::dedup_1_flavour<size_t>::policy policy_used = mondrian::vpipes::pipes::dedup_1_flavour<size_t>::policy::values;
    mondrian::vpipes::pipes::dedup_1_flavour<size_t> ded(&mat, batch_size, &ids, &vals, policy_used, true);
    testing_vpipes_classes::minimal_reader<size_t> reader(&mat, mondrian::vpipes::predicates::batched_predicates<size_t>
    ::greater_equal::micro_optimized_impl(0, true), loc_block_copy, input_length, batch_size, batch_size);

    reader.read();

    EXPECT_EQ(true, (res_length == original_res_length));
    free(result);
}