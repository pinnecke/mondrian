//
// Created by Mahmoud Mohsen on 21.05.17.
//

#pragma once

#include "gtest/gtest.h"
#include <testing_utilities.hpp>
#include <vpipes.hpp>

using namespace mondrian::vpipes;


/**
 * Test: test basic equi join between 2 columns of 20 elements
 *       rhs: contains elements from 0 to 20
 *       lhs: contains a single value 7 repeated 20 times
 *       All elements are non-null.
 *
 *  Right hand side operand
 *  +------------+------+------+
 *  | Tuplet id	 | val	| null |    Add to result
 *  +============+======+======+
 *  +- BATCH 1 --+------+------+
 *  | 0			 | 0	|      |    N
 *  | 1		     | 1	|      |    N
 *  | 2			 | 2	| 	   |    N
 *  | 3		     | 3	|      |    N
 *  | 4			 | 4	|      |    N
 *  | 5		     | 5	|      |    N
 *  | 6			 | 6	|      |    N
 *  | 7		     | 7    |      |    Y
 *  | 8			 | 8	|      |    N
 *  +- BATCH 2 --+------+------+
 *  | 9		     | 9	|      |    N
 *  | 10		 | 10	|      |    N
 *  | 11 		 | 11	|      |    N
 *  | 12		 | 12	|      |    N
 *  | 13		 | 13   |      |    N
 *  | 14		 | 14 	|      |    N
 *  | 15		 | 15   |      |    N
 *  | 16		 | 16	|      |    N
 *  | 17		 | 17  	|      |    N
 *  +- BATCH 3 --+------+------+
 *  | 18		 | 18	|      |    N
 *  | 19		 | 19   |      |    N
 *  +------------+------+------+
 *  Left hand side operand
 *  +------------+------+------+
 *  | Tuplet id	 | val	| null |    Add to result
 *  +============+======+======+
 *  +- BATCH 1 --+------+------+
 *  | 0			 | 7	|      |    Y
 *  | 1		     | 7	|      |    Y
 *  | 2			 | 7	| 	   |    Y
 *  | 3		     | 7	|      |    Y
 *  | 4			 | 7	|      |    Y
 *  | 5		     | 7	|      |    Y
 *  | 6			 | 7	|      |    Y
 *  | 7		     | 7    |      |    Y
 *  | 8			 | 7	|      |    Y
 *  +- BATCH 2 --+------+------+
 *  | 9		     | 7	|      |    Y
 *  | 10		 | 7	|      |    Y
 *  | 11 		 | 7 	|      |    Y
 *  | 12		 | 7	|      |    Y
 *  | 13		 | 7    |      |    Y
 *  | 14		 | 7 	|      |    Y
 *  | 15		 | 7    |      |    Y
 *  | 16		 | 7	|      |    Y
 *  | 17		 | 7   	|      |    Y
 *  +- BATCH 3 --+------+------+
 *  | 18		 | 7	|      |    Y
 *  | 19		 | 7    |      |    Y
 *  +------------+------+------+
 *
 *  The outcome should be
 *      3 batches of size 20 containing,
 *      Ids: all of the Ids of the left hand operand
 *      vals: it will take only one value 7 ... the ID which contains 7 in the right hand column
 */
TEST(TestStlHashJoin, TestEquiJoinNoneNull)
{
    std::cout << "\n\n>>> Test 'TestEquiJoinNoneNull'..." << std::endl;

    size_t ids_res_length = 1000;
    size_t vals_res_length = 1000;
    auto batch_size = 9;
    auto  input_length= 20;
    auto  val_result = new size_t[ids_res_length];
    auto  ids_result = new size_t[vals_res_length];
    auto predicate_value = 0 ;

    mondrian::vpipes::point_copy<size_t >::func_t ids_copier = [] (size_t *values, const size_t *tupletids, size_t num_of_ids)
    {
        for (auto i = 0; i < num_of_ids; ++i) {
            *(values + i) = *(tupletids + i);
        }
    };

    mondrian::vpipes::pipes::val_materialize<size_t> val_mat(val_result, &ids_res_length);
    mondrian::vpipes::pipes::tid_materialize<size_t> ids_mat(ids_result, &vals_res_length);

    interval<size_t> all_tuplet_ids(0, input_length);


    mondrian::vpipes::block_copy<size_t>::func_t loc_block_copy = [](size_t *values, size_t begin, size_t end){
        auto distance = end- begin;
        for (auto i = 0 ; i < distance ;++i){
            *(values + i) =begin + i;
        }
    };

    mondrian::vpipes::block_copy<size_t>::func_t loc_block_copy2 = [](size_t *values, size_t begin, size_t end){
        auto distance = end - begin;
        for (auto i = 0 ; i < distance; ++i){
            *(values + i) = 7;
        }
    };

    mondrian::vpipes::block_null_copy::func_t loc_block_null_copy = [] (mondrian::mtl::smart_bitmask *values,
                                                                        const mondrian::mtl::smart_array<size_t> *null_mask_indices,
                                                                        const mondrian::mtl::smart_array<tuplet_id_t> *tuplet_ids)
    {
        assert (values != nullptr);
        values->unset_range_safe(0, null_mask_indices->get_num_elements());
    };

    mondrian::vpipes::block_null_copy::func_t loc_block_null_copy2 = [] (mondrian::mtl::smart_bitmask *null_mask,
                                                                         const mondrian::mtl::smart_array<size_t> *null_mask_indices,
                                                                         const mondrian::mtl::smart_array<tuplet_id_t> *tuplet_ids)
    {
        for (int i = 0; i < null_mask_indices->get_num_elements(); i += 2) {
            null_mask->set_unsafe(i, true);
            assert (null_mask->get_unsafe(i));
        }
    };

    mondrian::vpipes::pipes::tee <size_t> tee_op(&val_mat, &ids_mat, batch_size);

    mondrian::vpipes::bi_pipes::stl_hash_join <size_t> stl_hash_join (&tee_op, batch_size);

    auto rhs_table = pipes::table_scan<size_t >(stl_hash_join.get_inner_operand(), &all_tuplet_ids, &all_tuplet_ids + 1,mondrian::vpipes::predicates::batched_predicates<size_t >
                                                ::greater_equal::micro_optimized_impl(predicate_value,true),
                                                null_value_filter_policy::skip_null_values,
                                                loc_block_copy, loc_block_null_copy, batch_size, batch_size, true);

    auto lhs_table = pipes::table_scan<size_t >(stl_hash_join.get_outer_operand(), &all_tuplet_ids, &all_tuplet_ids + 1,mondrian::vpipes::predicates::batched_predicates<size_t >
                                                ::greater_equal::micro_optimized_impl(predicate_value,true),
                                                null_value_filter_policy::skip_null_values,
                                                loc_block_copy2, loc_block_null_copy, batch_size, batch_size, true);


    rhs_table.start();
    lhs_table.start();

    auto expected_ids = new size_t [ids_res_length];
    auto expected_vals = new size_t [vals_res_length];
    std::iota (expected_ids, expected_ids + ids_res_length, 0);
    std::fill (expected_vals, expected_vals + vals_res_length, 7);

    EXPECT_EQ(20, stl_hash_join.get_join_statistics()->count_join_pairs);
    EXPECT_EQ(6, stl_hash_join.get_join_statistics()->count_non_null_branch_used);
    EXPECT_EQ(0, stl_hash_join.get_join_statistics()->count_null_values);
    EXPECT_EQ(0, stl_hash_join.get_join_statistics()->count_null_branch_used);
    EXPECT_EQ(true, has_same_vals(expected_ids, ids_result, ids_res_length));
    EXPECT_EQ(true, has_same_vals(expected_vals, val_result, vals_res_length));

    delete []val_result;
    delete []ids_result;
    delete []expected_ids;
    delete []expected_vals;
    stl_hash_join.clean_storage();
    rhs_table.dispose();
    lhs_table.dispose();
}

/**
 * Test: test basic equi join between a table of elements from 1 to 9
 *       the other table contains a single element 7
 *       All elements are non-null.
 *
 *  Right hand side operand
 *  +------------+------+------+
 *  | Tuplet id	 | val	| null |    Add to result
 *  +============+======+======+
 *  +- BATCH 1 --+------+------+
 *  | 0			 | 0	|      |    N
 *  | 1		     | 1	|      |    N
 *  | 2			 | 2	| 	   |    N
 *  | 3		     | 3	|      |    N
 *  | 4			 | 4	|      |    N
 *  | 5		     | 5	|      |    N
 *  | 6			 | 6	|      |    N
 *  | 7		     | 7    |      |    Y
 *  | 8			 | 8	|      |    N
 *  +- BATCH 2 --+------+------+
 *  | 9		     | 9	|      |    N
 *  | 10		 | 10	|      |    N
 *  | 11 		 | 11	|      |    N
 *  | 12		 | 12	|      |    N
 *  | 13		 | 13   |      |    N
 *  | 14		 | 14 	|      |    N
 *  | 15		 | 15   |      |    N
 *  | 16		 | 16	|      |    N
 *  | 17		 | 17  	|      |    N
 *  +- BATCH 3 --+------+------+
 *  | 18		 | 18	|      |    N
 *  | 19		 | 19   |      |    N
 *  +------------+------+------+
 *  Left hand side operand
 *  +------------+------+------+
 *  | Tuplet id	 | val	| null |    Add to result
 *  +============+======+======+
 *  +- BATCH 1 --+------+------+
 *  | 0			 | 7	|  y   |    N
 *  | 1		     | 7	|      |    Y
 *  | 2			 | 7	|  y   |    N
 *  | 3		     | 7	|      |    Y
 *  | 4			 | 7	|  y   |    N
 *  | 5		     | 7	|      |    Y
 *  | 6			 | 7	|  y   |    N
 *  | 7		     | 7    |      |    Y
 *  | 8			 | 7	|  y   |    N
 *  +- BATCH 2 --+------+------+
 *  | 9		     | 7	|  y   |    N
 *  | 10		 | 7	|      |    Y
 *  | 11 		 | 7 	|  y   |    N
 *  | 12		 | 7	|      |    Y
 *  | 13		 | 7    |  y   |    N
 *  | 14		 | 7 	|      |    Y
 *  | 15		 | 7    |  y   |    N
 *  | 16		 | 7	|      |    Y
 *  | 17		 | 7   	|  y   |    N
 *  +- BATCH 3 --+------+------+
 *  | 18		 | 7	|  y   |    N
 *  | 19		 | 7    |      |    Y
 *  +------------+------+------+
 *
 *  The outcome should be
 *      3 batches of size 20 containing,
 *      Ids: All of the Ids of the left hand operand that are not null, skip the null values
 *      vals: it will take only one value 7 ... the ID which contains 7 in the right hand column
 */
TEST(TestStlHashJoin, TestEquiJoinNull)
{
    std::cout << "\n\n>>> Test 'TestEquiJoinNull'..." << std::endl;

    size_t ids_res_length = 1000;
    size_t vals_res_length = 1000;
    auto batch_size = 9;
    auto  input_length= 20;
    auto  val_result = new size_t[ids_res_length];
    auto  ids_result = new size_t[vals_res_length];
    auto predicate_value = 0 ;

    mondrian::vpipes::point_copy<size_t >::func_t ids_copier = [] (size_t *values, const size_t *tupletids, size_t num_of_ids)
    {
        for (auto i = 0; i < num_of_ids; ++i) {
            *(values + i) = *(tupletids + i);
        }
    };

    mondrian::vpipes::pipes::val_materialize<size_t> val_mat(val_result, &ids_res_length);
    mondrian::vpipes::pipes::tid_materialize<size_t> ids_mat(ids_result, &vals_res_length);

    interval<size_t> all_tuplet_ids(0, input_length);


    mondrian::vpipes::block_copy<size_t>::func_t loc_block_copy = [](size_t *values, size_t begin, size_t end){
        auto distance = end- begin;
        for (auto i = 0 ; i < distance ;++i){
            *(values + i) =begin + i;
        }
    };

    mondrian::vpipes::block_copy<size_t>::func_t loc_block_copy2 = [](size_t *values, size_t begin, size_t end){
        auto distance = end - begin;
        for (auto i = 0 ; i < distance; ++i){
            *(values + i) = 7;
        }
    };

    mondrian::vpipes::block_null_copy::func_t loc_block_null_copy = [] (mondrian::mtl::smart_bitmask *values,
                                                                        const mondrian::mtl::smart_array<size_t> *null_mask_indices,
                                                                        const mondrian::mtl::smart_array<tuplet_id_t> *tuplet_ids)
    {
        assert (values != nullptr);
        values->unset_range_safe(0, null_mask_indices->get_num_elements());
    };

    mondrian::vpipes::block_null_copy::func_t loc_block_null_copy2 = [] (mondrian::mtl::smart_bitmask *null_mask,
                                                                         const mondrian::mtl::smart_array<size_t> *null_mask_indices,
                                                                         const mondrian::mtl::smart_array<tuplet_id_t> *tuplet_ids)
    {
        for (int i = 0; i < null_mask_indices->get_num_elements(); i += 2) {
            null_mask->set_unsafe(i, true);
            assert (null_mask->get_unsafe(i));
        }
    };

    mondrian::vpipes::pipes::tee <size_t> tee_op(&val_mat, &ids_mat, batch_size);

    mondrian::vpipes::bi_pipes::stl_hash_join<size_t> stl_hash_join (&tee_op,batch_size);

    auto rhs_table = pipes::table_scan<size_t >(stl_hash_join.get_inner_operand(), &all_tuplet_ids, &all_tuplet_ids + 1,mondrian::vpipes::predicates::batched_predicates<size_t >
                                                ::greater_equal::micro_optimized_impl(predicate_value,true),
                                                null_value_filter_policy::skip_null_values,
                                                loc_block_copy, loc_block_null_copy, batch_size, batch_size, true);

    auto lhs_table = pipes::table_scan<size_t >(stl_hash_join.get_outer_operand(), &all_tuplet_ids, &all_tuplet_ids + 1,mondrian::vpipes::predicates::batched_predicates<size_t >
                                                ::greater_equal::micro_optimized_impl(predicate_value,true),
                                                null_value_filter_policy::skip_null_values,
                                                loc_block_copy2, loc_block_null_copy2, batch_size, batch_size, true);


    rhs_table.start();
    lhs_table.start();
    std::vector<size_t> v{1, 3, 5, 7, 10, 12, 14, 16, 19};

    auto expected_vals = new size_t [vals_res_length];
    std::fill (expected_vals, expected_vals + vals_res_length, 7);

    EXPECT_EQ(9, stl_hash_join.get_join_statistics()->count_join_pairs);
    EXPECT_EQ(3, stl_hash_join.get_join_statistics()->count_non_null_branch_used);
    EXPECT_EQ(11, stl_hash_join.get_join_statistics()->count_null_values);
    EXPECT_EQ(3, stl_hash_join.get_join_statistics()->count_null_branch_used);
    EXPECT_EQ(true, has_same_vals(v.data(), ids_result, ids_res_length));
    EXPECT_EQ(true, has_same_vals(expected_vals, val_result, vals_res_length));

    delete []val_result;
    delete []ids_result;
    delete []expected_vals;
    stl_hash_join.clean_storage();
    rhs_table.dispose();
    lhs_table.dispose();
}


/**
 * Test: test basic equi join between two columns of different sizes
 *       rhs: contains 20 elements values from (0, 19)
 *       lhs: contains 9 elements from (0,9).
 *
 *  Right hand side operand
 *  +------------+------+------+
 *  | Tuplet id	 | val	| null |    Add to result
 *  +============+======+======+
 *  +- BATCH 1 --+------+------+
 *  | 0			 | 0	|      |    Y
 *  | 1		     | 1	|      |    Y
 *  | 2			 | 2	| 	   |    Y
 *  | 3		     | 3	|      |    Y
 *  | 4			 | 4	|      |    Y
 *  | 5		     | 5	|      |    Y
 *  | 6			 | 6	|      |    Y
 *  | 7		     | 7    |      |    Y
 *  | 8			 | 8	|      |    Y
 *  +- BATCH 2 --+------+------+
 *  | 9		     | 9	|      |    N
 *  | 10		 | 10	|      |    N
 *  | 11 		 | 11	|      |    N
 *  | 12		 | 12	|      |    N
 *  | 13		 | 13   |      |    N
 *  | 14		 | 14 	|      |    N
 *  | 15		 | 15   |      |    N
 *  | 16		 | 16	|      |    N
 *  | 17		 | 17  	|      |    N
 *  +- BATCH 3 --+------+------+
 *  | 18		 | 18	|      |    N
 *  | 19		 | 19   |      |    N
 *  +------------+------+------+
 *
 *  Left hand side operand
 *  +------------+------+------+
 *  | Tuplet id	 | val	| null |    Add to result
 *  +============+======+======+
 *  +- BATCH 1 --+------+------+
 *  | 0			 | 0	|      |    Y
 *  | 1		     | 1	|      |    Y
 *  | 2			 | 2	| 	   |    Y
 *  | 3		     | 3	|      |    Y
 *  | 4			 | 4	|      |    Y
 *  | 5		     | 5	|      |    Y
 *  | 6			 | 6	|      |    Y
 *  | 7		     | 7    |      |    Y
 *  | 8			 | 8	|      |    Y
 *  +------------+------+------+
 *
 *  The outcome should be
 *      1 batch contains elements from 0 to 8
 */
TEST(TestStlHashJoin, TestEquiJoinDiffLengthColumns)
{
    std::cout << "\n\n>>> Test 'TestEquiJoinDiffLengthColumns'..." << std::endl;

    size_t ids_res_length = 1000;
    size_t vals_res_length = 1000;
    auto batch_size = 9;
    auto  rhs_length = 20;
    auto  lhs_length = 9;
    auto  val_result = new size_t[ids_res_length];
    auto  ids_result = new size_t[vals_res_length];
    auto predicate_value = 0 ;

    mondrian::vpipes::point_copy<size_t >::func_t ids_copier = [] (size_t *values, const size_t *tupletids, size_t num_of_ids)
    {
        for (auto i = 0; i < num_of_ids; ++i) {
            *(values + i) = *(tupletids + i);
        }
    };

    mondrian::vpipes::pipes::val_materialize<size_t> val_mat(val_result, &ids_res_length);
    mondrian::vpipes::pipes::tid_materialize<size_t> ids_mat(ids_result, &vals_res_length);

    interval<size_t> all_tuplet_ids(0, rhs_length);
    interval<size_t> all_tuplet_ids2(0, lhs_length);

    mondrian::vpipes::block_copy<size_t>::func_t loc_block_copy = [](size_t *values, size_t begin, size_t end){
        auto distance = end- begin;
        for (auto i = 0 ; i < distance ;++i){
            *(values + i) =begin + i;
        }
    };

    mondrian::vpipes::block_copy<size_t>::func_t loc_block_copy2 = [](size_t *values, size_t begin, size_t end){
        auto distance = end - begin;
        for (auto i = 0 ; i < distance; ++i){
            *(values + i) = 7;
        }
    };

    mondrian::vpipes::block_null_copy::func_t loc_block_null_copy = [] (mondrian::mtl::smart_bitmask *values,
                                                                        const mondrian::mtl::smart_array<size_t> *null_mask_indices,
                                                                        const mondrian::mtl::smart_array<tuplet_id_t> *tuplet_ids)
    {
        assert (values != nullptr);
        values->unset_range_safe(0, null_mask_indices->get_num_elements());
    };

    mondrian::vpipes::block_null_copy::func_t loc_block_null_copy2 = [] (mondrian::mtl::smart_bitmask *null_mask,
                                                                         const mondrian::mtl::smart_array<size_t> *null_mask_indices,
                                                                         const mondrian::mtl::smart_array<tuplet_id_t> *tuplet_ids)
    {
        for (int i = 0; i < null_mask_indices->get_num_elements(); i += 2) {
            null_mask->set_unsafe(i, true);
            assert (null_mask->get_unsafe(i));
        }
    };

    mondrian::vpipes::pipes::tee <size_t> tee_op(&val_mat, &ids_mat, batch_size);

    mondrian::vpipes::bi_pipes::stl_hash_join<size_t> stl_hash_join (&tee_op,batch_size);

    auto rhs_table = pipes::table_scan<size_t >(stl_hash_join.get_inner_operand(), &all_tuplet_ids, &all_tuplet_ids + 1,mondrian::vpipes::predicates::batched_predicates<size_t >
                                                ::greater_equal::micro_optimized_impl(predicate_value,true),
                                                null_value_filter_policy::skip_null_values,
                                                loc_block_copy, loc_block_null_copy, batch_size, batch_size, true);

    auto lhs_table = pipes::table_scan<size_t >(stl_hash_join.get_outer_operand(), &all_tuplet_ids2, &all_tuplet_ids2 + 1,mondrian::vpipes::predicates::batched_predicates<size_t >
                                                ::greater_equal::micro_optimized_impl(predicate_value,true),
                                                null_value_filter_policy::skip_null_values,
                                                loc_block_copy, loc_block_null_copy, batch_size, batch_size, true);


    rhs_table.start();
    lhs_table.start();

    auto expected_ids = new size_t [ids_res_length];
    std::iota (expected_ids, expected_ids + ids_res_length, 0);

    EXPECT_EQ(9, stl_hash_join.get_join_statistics()->count_join_pairs);
    EXPECT_EQ(4, stl_hash_join.get_join_statistics()->count_non_null_branch_used);
    EXPECT_EQ(0, stl_hash_join.get_join_statistics()->count_null_values);
    EXPECT_EQ(0, stl_hash_join.get_join_statistics()->count_null_branch_used);
    EXPECT_EQ(true, has_same_vals(expected_ids, ids_result, ids_res_length));
    EXPECT_EQ(true, has_same_vals(expected_ids, val_result, vals_res_length));

    delete []val_result;
    delete []ids_result;
    delete []expected_ids;
    stl_hash_join.clean_storage();
    rhs_table.dispose();
    lhs_table.dispose();
}

