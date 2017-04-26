//
// Created by Marcus Pinnecke on 20.04.17.
//

#pragma once

#include "gtest/gtest.h"
#include <testing_utilities.hpp>
#include <vpipes.hpp>

using namespace mondrian::vpipes;

TEST(TestNullableOps, TestUseNullCheckingBranchStraightforwardFilter)
{
    using namespace mondrian::vpipes::pipes;
    using namespace mondrian::vpipes::predicates;
    using namespace mondrian::mtl;
    auto point_copy_fn = utilities::ids_copier;
    const size_t INPUT_DATA_LENGTH = 20, BATCH_SIZE = 9;

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
                                                     [&] (smart_bitmask *null_mask, const mondrian::mtl::smart_array<size_t> *null_mask_indices,
                                                          const mondrian::mtl::smart_array<tuplet_id_t> *tuplet_ids)
                                                     {
                                                     },
                                                     BATCH_SIZE, BATCH_SIZE, true);
    table_scan->start();
    auto builtin_filter = table_scan->get_filter();
    auto statistics = builtin_filter->get_predicate_statistics();

    EXPECT_EQ(result_buffer_length, INPUT_DATA_LENGTH);
    EXPECT_EQ(statistics->count_non_null_branch_used, std::ceil(INPUT_DATA_LENGTH / float(BATCH_SIZE)));
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
    const size_t INPUT_DATA_LENGTH = 500, BATCH_SIZE = 9;

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
                                                    [&] (smart_bitmask *null_mask, const mondrian::mtl::smart_array<size_t> *null_mask_indices,
                                                         const mondrian::mtl::smart_array<tuplet_id_t> *tuplet_ids)
                                                    {
                                                        null_mask->set_all();
                                                    },
                                                    BATCH_SIZE, BATCH_SIZE, true);
    table_scan->start();
    auto builtin_filter = table_scan->get_filter();
    auto statistics = builtin_filter->get_predicate_statistics();

    EXPECT_EQ(result_buffer_length, INPUT_DATA_LENGTH);
    EXPECT_EQ(statistics->count_non_null_branch_used, 0);
    EXPECT_EQ(statistics->count_null_branch_used, std::ceil(INPUT_DATA_LENGTH / float(BATCH_SIZE)));
    EXPECT_EQ(statistics->count_satisfying_values, 0);
    EXPECT_EQ(statistics->count_skipped_values, INPUT_DATA_LENGTH);
    EXPECT_EQ(statistics->count_non_satisfying_values, 0);
    EXPECT_EQ(statistics->count_null_values, INPUT_DATA_LENGTH);

    table_scan->dispose();
}

struct outcome
{
    size_t resultset_buffer_length;
    size_t scan_operator_statistics_out_num_batches;
    size_t scan_operator_statistics_out_num_empty_batches;
    size_t scan_operator_statistics_out_num_num_tuplets;
    size_t filt_operator_statistics_in_num_batches;
    size_t filt_operator_statistics_out_num_batches;
    size_t filt_operator_statistics_in_num_empty_batches;
    size_t filt_operator_statistics_out_num_empty_batches;
    size_t filt_operator_statistics_in_num_tuplets;
    size_t filt_operator_statistics_out_num_tuplets;
    size_t predicate_statistics_count_null_branch_used;
    size_t predicate_statistics_count_non_null_branch_used;
    size_t predicate_statistics_count_satisfying_values;
    size_t predicate_statistics_count_non_satisfying_values;
    size_t predicate_statistics_count_skipped_values;
    size_t predicate_statistics_count_null_values;
    size_t mat_operator_statistics_in_num_batches;
    size_t mat_operator_statistics_in_num_empty_batches;
    size_t mat_operator_statistics_in_num_num_tuplets;
};

outcome run_filter(size_t INPUT_DATA_LENGTH, size_t BATCH_SIZE,
                   mondrian::vpipes::predicates::batched_predicates<size_t>::func_t eval_impl,
                   mondrian::vpipes::pipes::table_scan<size_t>::block_null_copy_t null_copy_func,
                   null_value_filter_policy policy)
{
    using namespace mondrian::vpipes::pipes;
    using namespace mondrian::vpipes::predicates;
    using namespace mondrian::mtl;
    auto point_copy_fn = utilities::ids_copier;

    size_t *data = (size_t *) malloc (INPUT_DATA_LENGTH * sizeof(size_t));
    for (size_t i = 0; i < INPUT_DATA_LENGTH; ++i) {
        data[i] = i % 2;
    }

    bool *result_buffer = (bool *) malloc (INPUT_DATA_LENGTH * sizeof(bool));
    size_t result_buffer_length = 0;

    auto all_tuplet_ids = interval<size_t>(0, INPUT_DATA_LENGTH);

    nullmask_materialize<size_t> mat(result_buffer, &result_buffer_length);
    auto table_scan = new pipes::table_scan<size_t>(&mat, &all_tuplet_ids, &all_tuplet_ids + 1, eval_impl,
                                                    policy,
                                                    [&] (size_t *destination, tuplet_id_t begin, tuplet_id_t end)
                                                    {
                                                        memcpy(destination, data + begin, (end - begin) * sizeof(size_t));
                                                    },
                                                    null_copy_func,
                                                    BATCH_SIZE, BATCH_SIZE, true);
    table_scan->start();
    free (data);
    free (result_buffer);

    auto builtin_filter = table_scan->get_filter();

    auto predicate_statistics = builtin_filter->get_predicate_statistics();

    auto scan_operator_statistics_out = table_scan->get_output_statistics();

    auto filt_operator_statistics_in = builtin_filter->get_input_statistics();
    auto filt_operator_statistics_out = builtin_filter->get_output_statistics();

    auto mat_operator_statistics_in = mat.get_input_statistics();

    std::cout << "SCAN" << std::endl;
    std::cout << " |  #batches (in/out/in_empty/out_empty): " << "-/" << scan_operator_statistics_out->num_batches << "/-/" << scan_operator_statistics_out->num_empty_batches << std::endl;
    std::cout << " v  #tuplets (in/out): " << "-/" << scan_operator_statistics_out->num_tuplets << std::endl;
    std::cout << "FILTER" << std::endl;
    std::cout << " |  #batches (in/out/in_empty/out_empty): " << filt_operator_statistics_in->num_batches << "/" << filt_operator_statistics_out->num_batches << "/" << filt_operator_statistics_in->num_empty_batches << "/" << filt_operator_statistics_out->num_empty_batches << std::endl;
    std::cout << " |  #tuplets (in/out): " << filt_operator_statistics_in->num_tuplets << "/" << filt_operator_statistics_out->num_tuplets << std::endl;
    std::cout << " |  branching (null-branch/non-null-branch): " << predicate_statistics->count_null_branch_used << "/" << predicate_statistics->count_non_null_branch_used << std::endl;
    std::cout << " |  satisfaction (passed/removed/skipped): " << predicate_statistics->count_satisfying_values << "/" << predicate_statistics->count_non_satisfying_values << "/" << predicate_statistics->count_skipped_values << std::endl;
    std::cout << " v  #null values: " << predicate_statistics->count_null_values << std::endl;
    std::cout << "MATERIALIZE" << std::endl;
    std::cout << "    #batches (in/out/in_empty/out_empty): " << mat_operator_statistics_in->num_batches << "/" << "-" << "/" << mat_operator_statistics_in->num_empty_batches << "/" << "-" << std::endl;
    std::cout << "    #tuplets (in/out): " << mat_operator_statistics_in->num_tuplets << "/-" << std::endl;

    outcome result;
    result.resultset_buffer_length = result_buffer_length;
    result.scan_operator_statistics_out_num_batches = scan_operator_statistics_out->num_batches;
    result.scan_operator_statistics_out_num_empty_batches = scan_operator_statistics_out->num_empty_batches;
    result.scan_operator_statistics_out_num_num_tuplets = scan_operator_statistics_out->num_tuplets;
    result.filt_operator_statistics_in_num_batches = filt_operator_statistics_in->num_batches;
    result.filt_operator_statistics_out_num_batches = filt_operator_statistics_out->num_batches;
    result.filt_operator_statistics_in_num_empty_batches = filt_operator_statistics_in->num_empty_batches;
    result.filt_operator_statistics_out_num_empty_batches = filt_operator_statistics_out->num_empty_batches;
    result.filt_operator_statistics_in_num_tuplets = filt_operator_statistics_in->num_tuplets;
    result.filt_operator_statistics_out_num_tuplets = filt_operator_statistics_out->num_tuplets;
    result.predicate_statistics_count_null_branch_used = predicate_statistics->count_null_branch_used;
    result.predicate_statistics_count_non_null_branch_used = predicate_statistics->count_non_null_branch_used;
    result.predicate_statistics_count_satisfying_values = predicate_statistics->count_satisfying_values;
    result.predicate_statistics_count_non_satisfying_values = predicate_statistics->count_non_satisfying_values;
    result.predicate_statistics_count_skipped_values = predicate_statistics->count_skipped_values;
    result.predicate_statistics_count_null_values = predicate_statistics->count_null_values;
    result.mat_operator_statistics_in_num_batches = mat_operator_statistics_in->num_batches;
    result.mat_operator_statistics_in_num_empty_batches = mat_operator_statistics_in->num_empty_batches;
    result.mat_operator_statistics_in_num_num_tuplets = mat_operator_statistics_in->num_tuplets;

    table_scan->dispose();

    return result;
}

/**
 * Test: Passing null-masked untouched through the pipe using straightforward filter implementation that skip
 *       null values. All elements are non-null.
 *
 * With the following test-set, having a batch size of 9, a 100% selective filter and no deletion of nulls:
 *
 *  +------------+------+------+
 *  | Tuplet id	 | val	| null |    Add to result
 *  +============+======+======+
 *  +- BATCH 1 --+------+------+
 *  | 0			 | 0	|      |    Y
 *  | 1		     | 1	|      |    Y
 *  | 2			 | 0	| 	   |    Y
 *  | 3		     | 1	|      |    Y
 *  | 4			 | 0	|      |    Y
 *  | 5		     | 1	|      |    Y
 *  | 6			 | 0	|      |    Y
 *  | 7		     | 1    |      |    Y
 *  | 8			 | 0	|      |    Y
 *  +- BATCH 2 --+------+------+
 *  | 9		     | 1	|      |    Y
 *  | 10		 | 0	|      |    Y
 *  | 11 		 | 1 	|      |    Y
 *  | 12		 | 0	|      |    Y
 *  | 13		 | 1    |      |    Y
 *  | 14		 | 0 	|      |    Y
 *  | 15		 | 1    |      |    Y
 *  | 16		 | 0	|      |    Y
 *  | 17		 | 1   	|      |    Y
 *  +- BATCH 3 --+------+------+
 *  | 18		 | 0	|      |    Y
 *  | 19		 | 1    |      |    Y
 *  +------------+------+------+
 *
 *  The outcome should be
 *      -   3 batches should be created by table scan operator, passing 20 tuplets in total
 *      -   3 batches should be created by the filter operator, passing 20 values due to they satisfy the predicate
 *      -   3 batches should be materialized with a result buffer length of 20
 */
TEST(TestNullableOps, TestForwardNullMaskAllNonNullStraightforwardFilter)
{
    using namespace mondrian::vpipes::predicates;
    using namespace mondrian::mtl;

    std::cout << "\n\n>>> Test 'TestForwardNullMaskAllNonNullStraightforwardFilter'..." << std::endl;

    auto result = run_filter(20,    // Input data size
                             9,     // Batch size
                             batched_predicates<size_t>::greater_equal::straightforward_impl(0), // Predicate
                             [&] (smart_bitmask *null_mask, const mondrian::mtl::smart_array<size_t> *null_mask_indices,
                                  const mondrian::mtl::smart_array<tuplet_id_t> *tuplet_ids)
                             {

                             },
                             null_value_filter_policy::skip_null_values); // Null-Handling

    EXPECT_EQ(result.resultset_buffer_length, 20);
    EXPECT_EQ(result.scan_operator_statistics_out_num_batches, 3);
    EXPECT_EQ(result.scan_operator_statistics_out_num_empty_batches, 0);
    EXPECT_EQ(result.scan_operator_statistics_out_num_num_tuplets, 20);
    EXPECT_EQ(result.filt_operator_statistics_in_num_batches, 3);
    EXPECT_EQ(result.filt_operator_statistics_out_num_batches, 3);
    EXPECT_EQ(result.filt_operator_statistics_in_num_empty_batches, 0);
    EXPECT_EQ(result.filt_operator_statistics_out_num_empty_batches, 0);
    EXPECT_EQ(result.filt_operator_statistics_in_num_tuplets, 20);
    EXPECT_EQ(result.filt_operator_statistics_out_num_tuplets, 20);
    EXPECT_EQ(result.predicate_statistics_count_null_branch_used, 0);
    EXPECT_EQ(result.predicate_statistics_count_non_null_branch_used, 3);
    EXPECT_EQ(result.predicate_statistics_count_satisfying_values, 20);
    EXPECT_EQ(result.predicate_statistics_count_non_satisfying_values, 0);
    EXPECT_EQ(result.predicate_statistics_count_skipped_values, 0);
    EXPECT_EQ(result.predicate_statistics_count_null_values, 0);
    EXPECT_EQ(result.mat_operator_statistics_in_num_batches, 3);
    EXPECT_EQ(result.mat_operator_statistics_in_num_empty_batches, 0);
    EXPECT_EQ(result.mat_operator_statistics_in_num_num_tuplets, 20);
}

/**
 * Test: Passing null-masked untouched through the pipe using straightforward filter implementation that skip
 *       null values. All elements are null.
 *
 * With the following test-set, having a batch size of 9, a 100% selective filter and no deletion of nulls:
 *
 *  +------------+------+------+
 *  | Tuplet id	 | val	| null |    Add to result
 *  +============+======+======+
 *  +- BATCH 1 --+------+------+
 *  | 0			 | 0	|   Y  |    Y
 *  | 1		     | 1	|   Y  |    Y
 *  | 2			 | 0	| 	Y  |    Y
 *  | 3		     | 1	|   Y  |    Y
 *  | 4			 | 0	|   Y  |    Y
 *  | 5		     | 1	|   Y  |    Y
 *  | 6			 | 0	|   Y  |    Y
 *  | 7		     | 1    |   Y  |    Y
 *  | 8			 | 0	|   Y  |    Y
 *  +- BATCH 2 --+------+------+
 *  | 9		     | 1	|   Y  |    Y
 *  | 10		 | 0	|   Y  |    Y
 *  | 11 		 | 1 	|   Y  |    Y
 *  | 12		 | 0	|   Y  |    Y
 *  | 13		 | 1    |   Y  |    Y
 *  | 14		 | 0 	|   Y  |    Y
 *  | 15		 | 1    |   Y  |    Y
 *  | 16		 | 0	|   Y  |    Y
 *  | 17		 | 1   	|   Y  |    Y
 *  +- BATCH 3 --+------+------+
 *  | 18		 | 0	|   Y  |    Y
 *  | 19		 | 1    |   Y  |    Y
 *  +------------+------+------+
 *
 *  The outcome should be
 *      -   3 batches should be created by table scan operator, passing 20 tuplets in total
 *      -   3 batches should be created by the filter operator, passing 20 values due to they are null
 *      -   3 batches should be materialized with a result buffer length of 20
 */
TEST(TestNullableOps, TestForwardNullMaskAllNullStraightforwardFilter)
{
    using namespace mondrian::vpipes::predicates;
    using namespace mondrian::mtl;

    std::cout << "\n\n>>> Test 'TestForwardNullMaskAllNullStraightforwardFilter'..." << std::endl;

    auto result = run_filter(20,    // Input data size
                             9,     // Batch size
                             batched_predicates<size_t>::greater_equal::straightforward_impl(0), // Predicate
                             [&] (smart_bitmask *null_mask, const mondrian::mtl::smart_array<size_t> *null_mask_indices,
                                  const mondrian::mtl::smart_array<tuplet_id_t> *tuplet_ids)
                             {
                                 for (int i = 0; i < null_mask_indices->get_num_elements(); ++i) {
                                     null_mask->set_unsafe(i, true);
                                     assert (null_mask->get_unsafe(i));
                                 }
                             },
                             null_value_filter_policy::skip_null_values); // Null-Handling

    EXPECT_EQ(result.resultset_buffer_length, 20);
    EXPECT_EQ(result.scan_operator_statistics_out_num_batches, 3);
    EXPECT_EQ(result.scan_operator_statistics_out_num_empty_batches, 0);
    EXPECT_EQ(result.scan_operator_statistics_out_num_num_tuplets, 20);
    EXPECT_EQ(result.filt_operator_statistics_in_num_batches, 3);
    EXPECT_EQ(result.filt_operator_statistics_out_num_batches, 3);
    EXPECT_EQ(result.filt_operator_statistics_in_num_empty_batches, 0);
    EXPECT_EQ(result.filt_operator_statistics_out_num_empty_batches, 0);
    EXPECT_EQ(result.filt_operator_statistics_in_num_tuplets, 20);
    EXPECT_EQ(result.filt_operator_statistics_out_num_tuplets, 20);
    EXPECT_EQ(result.predicate_statistics_count_null_branch_used, 3);
    EXPECT_EQ(result.predicate_statistics_count_non_null_branch_used, 0);
    EXPECT_EQ(result.predicate_statistics_count_satisfying_values, 0);
    EXPECT_EQ(result.predicate_statistics_count_non_satisfying_values, 0);
    EXPECT_EQ(result.predicate_statistics_count_skipped_values, 20);
    EXPECT_EQ(result.predicate_statistics_count_null_values, 20);
    EXPECT_EQ(result.mat_operator_statistics_in_num_batches, 3);
    EXPECT_EQ(result.mat_operator_statistics_in_num_empty_batches, 0);
    EXPECT_EQ(result.mat_operator_statistics_in_num_num_tuplets, 20);
}

/**
 * Test: Passing null-masked untouched through the pipe using straightforward filter implementation that skip
 *       null values. Some elements are null.
 *
 * With the following test-set, having a batch size of 9, a 100% selective filter and no deletion of nulls:
 *
 *  +------------+------+------+
 *  | Tuplet id	 | val	| null |    Add to result
 *  +============+======+======+
 *  +- BATCH 1 --+------+------+
 *  | 0			 | 0	|  Y   |    Y
 *  | 1		     | 1	|      |    Y
 *  | 2			 | 0	| 	   |    Y
 *  | 3		     | 1	|  Y   |    Y
 *  | 4			 | 0	|      |    Y
 *  | 5		     | 1	|      |    Y
 *  | 6			 | 0	|  Y   |    Y
 *  | 7		     | 1    |      |    Y
 *  | 8			 | 0	|      |    Y
 *  +- BATCH 2 --+------+------+
 *  | 9		     | 1	|  Y   |    Y
 *  | 10		 | 0	|      |    Y
 *  | 11 		 | 1 	|      |    Y
 *  | 12		 | 0	|  Y   |    Y
 *  | 13		 | 1    |      |    Y
 *  | 14		 | 0 	|      |    Y
 *  | 15		 | 1    |  Y   |    Y
 *  | 16		 | 0	|      |    Y
 *  | 17		 | 1   	|      |    Y
 *  +- BATCH 3 --+------+------+
 *  | 18		 | 0	|  Y   |    Y
 *  | 19		 | 1    |      |    Y
 *  +------------+------+------+
 *
 *  The outcome should be
 *      -   3 batches should be created by table scan operator, passing 20 tuplets in total
 *      -   3 batches should be created by the filter operator, passing 7 values due to they satisfy the predicate + 13
 *                                                              which are null and passed anyway
 *      -   3 batches should be materialized with a result buffer length of 20
 */
TEST(TestNullableOps, TestForwardNullMaskSomeNullStraightforwardFilter)
{
    using namespace mondrian::vpipes::predicates;
    using namespace mondrian::mtl;

    std::cout << "\n\n>>> Test 'TestForwardNullMaskSomeNullStraightforwardFilter'..." << std::endl;

    auto result = run_filter(20,    // Input data size
                             9,     // Batch size
                             batched_predicates<size_t>::greater_equal::straightforward_impl(0), // Predicate
                             [&] (smart_bitmask *null_mask, const mondrian::mtl::smart_array<size_t> *null_mask_indices,
                                  const mondrian::mtl::smart_array<tuplet_id_t> *tuplet_ids)
                             {
                                 for (int i = 0; i < null_mask_indices->get_num_elements(); i += 3) {
                                     null_mask->set_unsafe(i, true);
                                     assert (null_mask->get_unsafe(i));
                                 }
                             },
                             null_value_filter_policy::skip_null_values); // Null-Handling

    EXPECT_EQ(result.resultset_buffer_length, 20);
    EXPECT_EQ(result.scan_operator_statistics_out_num_batches, 3);
    EXPECT_EQ(result.scan_operator_statistics_out_num_empty_batches, 0);
    EXPECT_EQ(result.scan_operator_statistics_out_num_num_tuplets, 20);
    EXPECT_EQ(result.filt_operator_statistics_in_num_batches, 3);
    EXPECT_EQ(result.filt_operator_statistics_out_num_batches, 3);
    EXPECT_EQ(result.filt_operator_statistics_in_num_empty_batches, 0);
    EXPECT_EQ(result.filt_operator_statistics_out_num_empty_batches, 0);
    EXPECT_EQ(result.filt_operator_statistics_in_num_tuplets, 20);
    EXPECT_EQ(result.filt_operator_statistics_out_num_tuplets, 20);
    EXPECT_EQ(result.predicate_statistics_count_null_branch_used, 3);
    EXPECT_EQ(result.predicate_statistics_count_non_null_branch_used, 0);
    EXPECT_EQ(result.predicate_statistics_count_satisfying_values, 13);
    EXPECT_EQ(result.predicate_statistics_count_non_satisfying_values, 0);
    EXPECT_EQ(result.predicate_statistics_count_skipped_values, 7);
    EXPECT_EQ(result.predicate_statistics_count_null_values, 7);
    EXPECT_EQ(result.mat_operator_statistics_in_num_batches, 3);
    EXPECT_EQ(result.mat_operator_statistics_in_num_empty_batches, 0);
    EXPECT_EQ(result.mat_operator_statistics_in_num_num_tuplets, 20);
}

///---------------------------------------------------------------------------------------------------------------------

/**
 * Test: Passing null-masked untouched through the pipe using straightforward filter implementation that removes
 *       null values. All elements are non-null.
 *
 * With the following test-set, having a batch size of 9, a 100% selective filter and no deletion of nulls:
 *
 *  +------------+------+------+
 *  | Tuplet id	 | val	| null |    Add to result
 *  +============+======+======+
 *  +- BATCH 1 --+------+------+
 *  | 0			 | 0	|      |    Y
 *  | 1		     | 1	|      |    Y
 *  | 2			 | 0	| 	   |    Y
 *  | 3		     | 1	|      |    Y
 *  | 4			 | 0	|      |    Y
 *  | 5		     | 1	|      |    Y
 *  | 6			 | 0	|      |    Y
 *  | 7		     | 1    |      |    Y
 *  | 8			 | 0	|      |    Y
 *  +- BATCH 2 --+------+------+
 *  | 9		     | 1	|      |    Y
 *  | 10		 | 0	|      |    Y
 *  | 11 		 | 1 	|      |    Y
 *  | 12		 | 0	|      |    Y
 *  | 13		 | 1    |      |    Y
 *  | 14		 | 0 	|      |    Y
 *  | 15		 | 1    |      |    Y
 *  | 16		 | 0	|      |    Y
 *  | 17		 | 1   	|      |    Y
 *  +- BATCH 3 --+------+------+
 *  | 18		 | 0	|      |    Y
 *  | 19		 | 1    |      |    Y
 *  +------------+------+------+
 *
 *  The outcome should be
 *      -   3 batches should be created by table scan operator, passing 20 tuplets in total
 *      -   3 batches should be created by the filter operator, passing 20 values due to they satisfy the predicate
 *      -   3 batches should be materialized with a result buffer length of 20
 */
TEST(TestNullableOps, TestRemoveNullMaskAllNonNullStraightforwardFilter)
{
    using namespace mondrian::vpipes::predicates;
    using namespace mondrian::mtl;

    std::cout << "\n\n>>> Test 'TestRemoveNullMaskAllNonNullStraightforwardFilter'..." << std::endl;

    auto result = run_filter(20,    // Input data size
                             9,     // Batch size
                             batched_predicates<size_t>::greater_equal::straightforward_impl(0), // Predicate
                             [&] (smart_bitmask *null_mask, const mondrian::mtl::smart_array<size_t> *null_mask_indices,
                                  const mondrian::mtl::smart_array<tuplet_id_t> *tuplet_ids)
                             {

                             },
                             null_value_filter_policy::remove_null_values); // Null-Handling

    EXPECT_EQ(result.resultset_buffer_length, 20);
    EXPECT_EQ(result.scan_operator_statistics_out_num_batches, 3);
    EXPECT_EQ(result.scan_operator_statistics_out_num_empty_batches, 0);
    EXPECT_EQ(result.scan_operator_statistics_out_num_num_tuplets, 20);
    EXPECT_EQ(result.filt_operator_statistics_in_num_batches, 3);
    EXPECT_EQ(result.filt_operator_statistics_out_num_batches, 3);
    EXPECT_EQ(result.filt_operator_statistics_in_num_empty_batches, 0);
    EXPECT_EQ(result.filt_operator_statistics_out_num_empty_batches, 0);
    EXPECT_EQ(result.filt_operator_statistics_in_num_tuplets, 20);
    EXPECT_EQ(result.filt_operator_statistics_out_num_tuplets, 20);
    EXPECT_EQ(result.predicate_statistics_count_null_branch_used, 0);
    EXPECT_EQ(result.predicate_statistics_count_non_null_branch_used, 3);
    EXPECT_EQ(result.predicate_statistics_count_satisfying_values, 20);
    EXPECT_EQ(result.predicate_statistics_count_non_satisfying_values, 0);
    EXPECT_EQ(result.predicate_statistics_count_skipped_values, 0);
    EXPECT_EQ(result.predicate_statistics_count_null_values, 0);
    EXPECT_EQ(result.mat_operator_statistics_in_num_batches, 3);
    EXPECT_EQ(result.mat_operator_statistics_in_num_empty_batches, 0);
    EXPECT_EQ(result.mat_operator_statistics_in_num_num_tuplets, 20);
}

/**
 * Test: Passing null-masked untouched through the pipe using straightforward filter implementation that removes
 *       null values. All elements are null.
 *
 * With the following test-set, having a batch size of 9, a 100% selective filter and no deletion of nulls:
 *
 *  +------------+------+------+
 *  | Tuplet id	 | val	| null |    Add to result
 *  +============+======+======+
 *  +- BATCH 1 --+------+------+
 *  | 0			 | 0	|   Y  |    N
 *  | 1		     | 1	|   Y  |    N
 *  | 2			 | 0	| 	Y  |    N
 *  | 3		     | 1	|   Y  |    N
 *  | 4			 | 0	|   Y  |    N
 *  | 5		     | 1	|   Y  |    N
 *  | 6			 | 0	|   Y  |    N
 *  | 7		     | 1    |   Y  |    N
 *  | 8			 | 0	|   Y  |    N
 *  +- BATCH 2 --+------+------+
 *  | 9		     | 1	|   Y  |    N
 *  | 10		 | 0	|   Y  |    N
 *  | 11 		 | 1 	|   Y  |    N
 *  | 12		 | 0	|   Y  |    N
 *  | 13		 | 1    |   Y  |    N
 *  | 14		 | 0 	|   Y  |    N
 *  | 15		 | 1    |   Y  |    N
 *  | 16		 | 0	|   Y  |    N
 *  | 17		 | 1   	|   Y  |    N
 *  +- BATCH 3 --+------+------+
 *  | 18		 | 0	|   Y  |    N
 *  | 19		 | 1    |   Y  |    N
 *  +------------+------+------+
 *
 *  The outcome should be
 *      -   3 batches should be created by table scan operator, passing 20 tuplets in total
 *      -   0 batches should be created by the filter operator, passing 0 values due to they are null
 *      -   0 batches should be materialized with a result buffer length of 0
 */
TEST(TestNullableOps, TestRemoveNullMaskAllNullStraightforwardFilter)
{
    using namespace mondrian::vpipes::predicates;
    using namespace mondrian::mtl;

    std::cout << "\n\n>>> Test 'TestRemoveNullMaskAllNullStraightforwardFilter'..." << std::endl;

    auto result = run_filter(20,    // Input data size
                             9,     // Batch size
                             batched_predicates<size_t>::greater_equal::straightforward_impl(0), // Predicate
                             [&] (smart_bitmask *null_mask, const mondrian::mtl::smart_array<size_t> *null_mask_indices,
                                  const mondrian::mtl::smart_array<tuplet_id_t> *tuplet_ids)
                             {
                                 for (int i = 0; i < null_mask_indices->get_num_elements(); ++i) {
                                     null_mask->set_unsafe(i, true);
                                     assert (null_mask->get_unsafe(i));
                                 }
                             },
                             null_value_filter_policy::remove_null_values); // Null-Handling

    EXPECT_EQ(result.resultset_buffer_length, 0);
    EXPECT_EQ(result.scan_operator_statistics_out_num_batches, 3);
    EXPECT_EQ(result.scan_operator_statistics_out_num_empty_batches, 0);
    EXPECT_EQ(result.scan_operator_statistics_out_num_num_tuplets, 20);
    EXPECT_EQ(result.filt_operator_statistics_in_num_batches, 3);
    EXPECT_EQ(result.filt_operator_statistics_out_num_batches, 0);
    EXPECT_EQ(result.filt_operator_statistics_in_num_empty_batches, 0);
    EXPECT_EQ(result.filt_operator_statistics_out_num_empty_batches, 0);
    EXPECT_EQ(result.filt_operator_statistics_in_num_tuplets, 20);
    EXPECT_EQ(result.filt_operator_statistics_out_num_tuplets, 0);
    EXPECT_EQ(result.predicate_statistics_count_null_branch_used, 3);
    EXPECT_EQ(result.predicate_statistics_count_non_null_branch_used, 0);
    EXPECT_EQ(result.predicate_statistics_count_satisfying_values, 0);
    EXPECT_EQ(result.predicate_statistics_count_non_satisfying_values, 0);
    EXPECT_EQ(result.predicate_statistics_count_skipped_values, 0);
    EXPECT_EQ(result.predicate_statistics_count_null_values, 20);
    EXPECT_EQ(result.mat_operator_statistics_in_num_batches, 0);
    EXPECT_EQ(result.mat_operator_statistics_in_num_empty_batches, 0);
    EXPECT_EQ(result.mat_operator_statistics_in_num_num_tuplets, 0);
}

/**
 * Test: Passing null-masked untouched through the pipe using straightforward filter implementation that removes
 *       null values. Some elements are null.
 *
 * With the following test-set, having a batch size of 9, a 100% selective filter and no deletion of nulls:
 *
 *  +------------+------+------+
 *  | Tuplet id	 | val	| null |    Add to result
 *  +============+======+======+
 *  +- BATCH 1 --+------+------+
 *  | 0			 | 0	|  Y   |    N
 *  | 1		     | 1	|      |    Y
 *  | 2			 | 0	| 	   |    Y
 *  | 3		     | 1	|  Y   |    N
 *  | 4			 | 0	|      |    Y
 *  | 5		     | 1	|      |    Y
 *  | 6			 | 0	|  Y   |    N
 *  | 7		     | 1    |      |    Y
 *  | 8			 | 0	|      |    Y
 *  +- BATCH 2 --+------+------+
 *  | 9		     | 1	|  Y   |    N
 *  | 10		 | 0	|      |    Y
 *  | 11 		 | 1 	|      |    Y
 *  | 12		 | 0	|  Y   |    N
 *  | 13		 | 1    |      |    Y
 *  | 14		 | 0 	|      |    Y
 *  | 15		 | 1    |  Y   |    N
 *  | 16		 | 0	|      |    Y
 *  | 17		 | 1   	|      |    Y
 *  +- BATCH 3 --+------+------+
 *  | 18		 | 0	|  Y   |    N
 *  | 19		 | 1    |      |    Y
 *  +------------+------+------+
 *
 *  The outcome should be
 *      -   3 batches should be created by table scan operator, passing 20 tuplets in total
 *      -   2 batches should be created by the filter operator, passing 13 values due to they satisfy and are non-null
 *      -   2 batches should be materialized with a result buffer length of 13
 */
TEST(TestNullableOps, TestRemoveNullMaskSomeNullStraightforwardFilter)
{
    using namespace mondrian::vpipes::predicates;
    using namespace mondrian::mtl;

    std::cout << "\n\n>>> Test 'TestRemoveNullMaskSomeNullStraightforwardFilter'..." << std::endl;

    auto result = run_filter(20,    // Input data size
                             9,     // Batch size
                             batched_predicates<size_t>::greater_equal::straightforward_impl(0), // Predicate
                             [&] (smart_bitmask *null_mask, const mondrian::mtl::smart_array<size_t> *null_mask_indices,
                                  const mondrian::mtl::smart_array<tuplet_id_t> *tuplet_ids)
                             {
                                 for (int i = 0; i < null_mask_indices->get_num_elements(); i += 3) {
                                     null_mask->set_unsafe(i, true);
                                     assert (null_mask->get_unsafe(i));
                                 }
                             },
                             null_value_filter_policy::remove_null_values); // Null-Handling

    EXPECT_EQ(result.resultset_buffer_length, 13);
    EXPECT_EQ(result.scan_operator_statistics_out_num_batches, 3);
    EXPECT_EQ(result.scan_operator_statistics_out_num_empty_batches, 0);
    EXPECT_EQ(result.scan_operator_statistics_out_num_num_tuplets, 20);
    EXPECT_EQ(result.filt_operator_statistics_in_num_batches, 3);
    EXPECT_EQ(result.filt_operator_statistics_out_num_batches, 2);
    EXPECT_EQ(result.filt_operator_statistics_in_num_empty_batches, 0);
    EXPECT_EQ(result.filt_operator_statistics_out_num_empty_batches, 0);
    EXPECT_EQ(result.filt_operator_statistics_in_num_tuplets, 20);
    EXPECT_EQ(result.filt_operator_statistics_out_num_tuplets, 13);
    EXPECT_EQ(result.predicate_statistics_count_null_branch_used, 3);
    EXPECT_EQ(result.predicate_statistics_count_non_null_branch_used, 0);
    EXPECT_EQ(result.predicate_statistics_count_satisfying_values, 13);
    EXPECT_EQ(result.predicate_statistics_count_non_satisfying_values, 0);
    EXPECT_EQ(result.predicate_statistics_count_skipped_values, 0);
    EXPECT_EQ(result.predicate_statistics_count_null_values, 7);
    EXPECT_EQ(result.mat_operator_statistics_in_num_batches, 2);
    EXPECT_EQ(result.mat_operator_statistics_in_num_empty_batches, 0);
    EXPECT_EQ(result.mat_operator_statistics_in_num_num_tuplets, 13);
}

/**
 * Test: Passing null-masked untouched through the pipe using straightforward filter implementation that removes
 *       null values. Some elements are null.
 *
 * With the following test-set, having a batch size of 9, a non-100% selective filter and no deletion of nulls:
 *
 *  +------------+--------------+------+
 *  | Tuplet id	 | val (match)	| null |    Add to result
 *  +============+==============+======+
 *  +- BATCH 1 --+--------------+------+
 *  | 0			 | 0 (N)    	|  Y   |    N
 *  | 1		     | 1 (Y)    	|      |    Y
 *  | 2			 | 0 (N)    	| 	   |    N
 *  | 3		     | 1 (Y)    	|  Y   |    N
 *  | 4			 | 0 (N)    	|      |    N
 *  | 5		     | 1 (Y)    	|      |    Y
 *  | 6			 | 0 (N)    	|  Y   |    N
 *  | 7		     | 1 (Y)    	|      |    Y
 *  | 8			 | 0 (N)    	|      |    N
 *  +- BATCH 2 --+--------------+------+
 *  | 9		     | 1 (Y)    	|  Y   |    N
 *  | 10		 | 0 (N)    	|      |    N
 *  | 11 		 | 1 (Y)    	|      |    Y
 *  | 12		 | 0 (N)    	|  Y   |    N
 *  | 13		 | 1 (Y)    	|      |    Y
 *  | 14		 | 0 (N)    	|      |    N
 *  | 15		 | 1 (Y)    	|  Y   |    N
 *  | 16		 | 0 (N)    	|      |    N
 *  | 17		 | 1 (Y)    	|      |    Y
 *  +- BATCH 3 --+------+------+
 *  | 18		 | 0 (N)    	|  Y   |    N
 *  | 19		 | 1 (Y)    	|      |    Y
 *  +------------+------+------+
 *
 *  The outcome should be
 *      -   3 batches should be created by table scan operator, passing 20 tuplets in total
 *      -   2 batches should be created by the filter operator, passing 13 values due to they satisfy and are non-null
 *      -   2 batches should be materialized with a result buffer length of 13
 */
TEST(TestNullableOps, TestComplexRemoveNullStraightforwardFilter)
{
    using namespace mondrian::vpipes::predicates;
    using namespace mondrian::mtl;

    std::cout << "\n\n>>> Test 'TestComplexRemoveNullStraightforwardFilter'..." << std::endl;

    auto result = run_filter(200,    // Input data size
                             66,     // Batch size
                             batched_predicates<size_t>::greater_equal::straightforward_impl(1), // Predicate
                             [&] (smart_bitmask *null_mask, const mondrian::mtl::smart_array<size_t> *null_mask_indices,
                                  const mondrian::mtl::smart_array<tuplet_id_t> *tuplet_ids)
                             {
                                 for (int i = 0; i < null_mask_indices->get_num_elements(); i += 3) {
                                     null_mask->set_unsafe(i, true);
                                     assert (null_mask->get_unsafe(i));
                                 }
                             },
                             null_value_filter_policy::remove_null_values); // Null-Handling

    EXPECT_EQ(result.resultset_buffer_length, 67);
    EXPECT_EQ(result.scan_operator_statistics_out_num_batches, 4);
    EXPECT_EQ(result.scan_operator_statistics_out_num_empty_batches, 0);
    EXPECT_EQ(result.scan_operator_statistics_out_num_num_tuplets, 200);
    EXPECT_EQ(result.filt_operator_statistics_in_num_batches, 4);
    EXPECT_EQ(result.filt_operator_statistics_out_num_batches, 2);
    EXPECT_EQ(result.filt_operator_statistics_in_num_empty_batches, 0);
    EXPECT_EQ(result.filt_operator_statistics_out_num_empty_batches, 0);
    EXPECT_EQ(result.filt_operator_statistics_in_num_tuplets, 200);
    EXPECT_EQ(result.filt_operator_statistics_out_num_tuplets, 67);
    EXPECT_EQ(result.predicate_statistics_count_null_branch_used, 4);
    EXPECT_EQ(result.predicate_statistics_count_non_null_branch_used, 0);
    EXPECT_EQ(result.predicate_statistics_count_satisfying_values, 67);
    EXPECT_EQ(result.predicate_statistics_count_non_satisfying_values, 66);
    EXPECT_EQ(result.predicate_statistics_count_skipped_values, 0);
    EXPECT_EQ(result.predicate_statistics_count_null_values, 67);
    EXPECT_EQ(result.mat_operator_statistics_in_num_batches, 2);
    EXPECT_EQ(result.mat_operator_statistics_in_num_empty_batches, 0);
    EXPECT_EQ(result.mat_operator_statistics_in_num_num_tuplets, 67);
}

/**
 * Test: Passing null-masked untouched through the pipe using straightforward filter implementation that passes
 *       null values. Some elements are null.
 *
 * With the following test-set, having a batch size of 9, a non-100% selective filter and no deletion of nulls:
 *
 *  +------------+--------------+------+
 *  | Tuplet id	 | val (match)	| null |    Add to result
 *  +============+==============+======+
 *  +- BATCH 1 --+--------------+------+
 *  | 0			 | 0 (N)    	|  Y   |    Y
 *  | 1		     | 1 (Y)    	|      |    Y
 *  | 2			 | 0 (N)    	| 	   |    N
 *  | 3		     | 1 (Y)    	|  Y   |    Y
 *  | 4			 | 0 (N)    	|      |    N
 *  | 5		     | 1 (Y)    	|      |    Y
 *  | 6			 | 0 (N)    	|  Y   |    Y
 *  | 7		     | 1 (Y)    	|      |    Y
 *  | 8			 | 0 (N)    	|      |    N
 *  +- BATCH 2 --+--------------+------+
 *  | 9		     | 1 (Y)    	|  Y   |    Y
 *  | 10		 | 0 (N)    	|      |    N
 *  | 11 		 | 1 (Y)    	|      |    Y
 *  | 12		 | 0 (N)    	|  Y   |    Y
 *  | 13		 | 1 (Y)    	|      |    Y
 *  | 14		 | 0 (N)    	|      |    N
 *  | 15		 | 1 (Y)    	|  Y   |    Y
 *  | 16		 | 0 (N)    	|      |    N
 *  | 17		 | 1 (Y)    	|      |    Y
 *  +- BATCH 3 --+------+------+
 *  | 18		 | 0 (N)    	|  Y   |    Y
 *  | 19		 | 1 (Y)    	|      |    Y
 *  +------------+------+------+
 *
 *  The outcome should be
 *      -   3 batches should be created by table scan operator, passing 20 tuplets in total
 *      -   2 batches should be created by the filter operator, passing 13 values due to they satisfy and are non-null
 *      -   2 batches should be materialized with a result buffer length of 13
 */
TEST(TestNullableOps, TestComplexPassNullStraightforwardFilter)
{
    using namespace mondrian::vpipes::predicates;
    using namespace mondrian::mtl;

    std::cout << "\n\n>>> Test 'TestComplexPassNullStraightforwardFilter'..." << std::endl;

    auto result = run_filter(20,    // Input data size
                             9,     // Batch size
                             batched_predicates<size_t>::greater_equal::straightforward_impl(1), // Predicate
                             [&] (smart_bitmask *null_mask, const mondrian::mtl::smart_array<size_t> *null_mask_indices,
                                  const mondrian::mtl::smart_array<tuplet_id_t> *tuplet_ids)
                             {
                                 for (int i = 0; i < null_mask_indices->get_num_elements(); i += 3) {
                                     null_mask->set_unsafe(i, true);
                                     assert (null_mask->get_unsafe(i));
                                 }
                             },
                             null_value_filter_policy::skip_null_values); // Null-Handling

    EXPECT_EQ(result.resultset_buffer_length, 14);
    EXPECT_EQ(result.scan_operator_statistics_out_num_batches, 3);
    EXPECT_EQ(result.scan_operator_statistics_out_num_empty_batches, 0);
    EXPECT_EQ(result.scan_operator_statistics_out_num_num_tuplets, 20);
    EXPECT_EQ(result.filt_operator_statistics_in_num_batches, 3);
    EXPECT_EQ(result.filt_operator_statistics_out_num_batches, 2);
    EXPECT_EQ(result.filt_operator_statistics_in_num_empty_batches, 0);
    EXPECT_EQ(result.filt_operator_statistics_out_num_empty_batches, 0);
    EXPECT_EQ(result.filt_operator_statistics_in_num_tuplets, 20);
    EXPECT_EQ(result.filt_operator_statistics_out_num_tuplets, 14);
    EXPECT_EQ(result.predicate_statistics_count_null_branch_used, 3);
    EXPECT_EQ(result.predicate_statistics_count_non_null_branch_used, 0);
    EXPECT_EQ(result.predicate_statistics_count_satisfying_values, 7);
    EXPECT_EQ(result.predicate_statistics_count_non_satisfying_values, 6);
    EXPECT_EQ(result.predicate_statistics_count_skipped_values, 7);
    EXPECT_EQ(result.predicate_statistics_count_null_values, 7);
    EXPECT_EQ(result.mat_operator_statistics_in_num_batches, 2);
    EXPECT_EQ(result.mat_operator_statistics_in_num_empty_batches, 0);
    EXPECT_EQ(result.mat_operator_statistics_in_num_num_tuplets, 14);
}