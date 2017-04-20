//
// Created by Marcus Pinnecke on 20.04.17.
//

#pragma once

#include "gtest/gtest.h"
#include <vpipes.hpp>

using namespace mondrian::vpipes;

TEST(TestNullableOps, TestUseNullCheckingBranchStraightforwardFilter)
{
    using namespace mondrian::vpipes::pipes;
    using namespace mondrian::vpipes::predicates;
    using namespace mondrian::mtl;
    auto point_copy_fn = utilities::ids_copier;
    const size_t INPUT_DATA_LENGTH = 500, BATCH_SIZE = 10;

    size_t *data = (size_t *) calloc (INPUT_DATA_LENGTH, sizeof(size_t));
    bool *result_buffer = (bool *) malloc (INPUT_DATA_LENGTH * sizeof(bool));
    size_t result_buffer_length = 0;

    auto eval_impl = batched_predicates<size_t>::greater_equal::straightforward_impl(0);
    auto all_tuplet_ids = interval<size_t>(0, INPUT_DATA_LENGTH);

    nullmask_materialize<size_t> mat(result_buffer, &result_buffer_length);
    auto table_scan = new pipes::table_scan<size_t>(&mat, &all_tuplet_ids, &all_tuplet_ids + 1, eval_impl,
                                                    null_value_filter_policy::skip_null_values,
                                                     [&] (size_t *destination, tuplet_id_t begin, tuplet_id_t end)
                                                     {
                                                         memcpy(destination, data + begin, (end - begin) * sizeof(size_t));
                                                     },
                                                     [&] (smart_bitmask *null_mask, tuplet_id_t begin, tuplet_id_t end)
                                                     {
                                                     },
                                                     BATCH_SIZE, BATCH_SIZE, true);
    table_scan->start();
    auto builtin_filter = table_scan->get_filter();
    auto statistics = builtin_filter->get_predicate_statistics();

    EXPECT_EQ(result_buffer_length, INPUT_DATA_LENGTH);
    EXPECT_EQ(statistics->count_non_null_branch_used, INPUT_DATA_LENGTH / BATCH_SIZE);
    EXPECT_EQ(statistics->count_null_branch_used, 0);
    EXPECT_EQ(statistics->count_skipped_values, 0);
    EXPECT_EQ(statistics->count_satisfying_values, INPUT_DATA_LENGTH);
    EXPECT_EQ(statistics->count_non_satisfying_values, 0);
    EXPECT_EQ(statistics->count_null_values, 0);

    table_scan->dispose();
}

TEST(TestNullableOps, TestSkipNullCheckingBranchStraightforwardFilter)
{
    using namespace mondrian::vpipes::pipes;
    using namespace mondrian::vpipes::predicates;
    using namespace mondrian::mtl;
    auto point_copy_fn = utilities::ids_copier;
    const size_t INPUT_DATA_LENGTH = 500, BATCH_SIZE = 10;

    size_t *data = (size_t *) calloc (INPUT_DATA_LENGTH, sizeof(size_t));
    bool *result_buffer = (bool *) malloc (INPUT_DATA_LENGTH * sizeof(bool));
    size_t result_buffer_length = 0;

    auto eval_impl = batched_predicates<size_t>::greater_equal::straightforward_impl(0);
    auto all_tuplet_ids = interval<size_t>(0, INPUT_DATA_LENGTH);

    nullmask_materialize<size_t> mat(result_buffer, &result_buffer_length);
    auto table_scan = new pipes::table_scan<size_t>(&mat, &all_tuplet_ids, &all_tuplet_ids + 1, eval_impl,
                                                    null_value_filter_policy::skip_null_values,
                                                    [&] (size_t *destination, tuplet_id_t begin, tuplet_id_t end)
                                                    {
                                                        memcpy(destination, data + begin, (end - begin) * sizeof(size_t));
                                                    },
                                                    [&] (smart_bitmask *null_mask, tuplet_id_t begin, tuplet_id_t end)
                                                    {
                                                        null_mask->set_all();
                                                    },
                                                    BATCH_SIZE, BATCH_SIZE, true);
    table_scan->start();
    auto builtin_filter = table_scan->get_filter();
    auto statistics = builtin_filter->get_predicate_statistics();

    EXPECT_EQ(result_buffer_length, INPUT_DATA_LENGTH);
    EXPECT_EQ(statistics->count_non_null_branch_used, 0);
    EXPECT_EQ(statistics->count_null_branch_used, INPUT_DATA_LENGTH / BATCH_SIZE);
    EXPECT_EQ(statistics->count_satisfying_values, 0);
    EXPECT_EQ(statistics->count_skipped_values, INPUT_DATA_LENGTH);
    EXPECT_EQ(statistics->count_non_satisfying_values, 0);
    EXPECT_EQ(statistics->count_null_values, INPUT_DATA_LENGTH);

    table_scan->dispose();
}
