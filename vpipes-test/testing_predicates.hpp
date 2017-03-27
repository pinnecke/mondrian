//
// Created by Mahmoud Mohsen on 3/15/17.
//

#pragma once

#include "gtest/gtest.h"
#include <vpipes.hpp>
#include <testing_utilities.hpp>

using namespace mondrian::vpipes;



TEST(TestStraightForwardImp,TestGreaterEqual ){
    size_t res_size =150 ;
    size_t input_size =150 ;
    auto ids = create_column(input_size,false);
    auto vals = create_column(input_size,false);
    auto result = create_column(res_size,true);
    size_t pred_val = 5 ;
    auto expected_result = generate_expected_result(input_size, [pred_val] (size_t val) -> size_t {
        return ! (val >=pred_val);
    });
    auto predicate=   mondrian::vpipes::predicates::batched_predicates<size_t>::greater_equal::straightforward_impl(pred_val);

    predicate(result, &res_size,
              ids, vals, input_size);

    EXPECT_EQ(has_same_vals(result,expected_result,res_size),true );
    delete_column(expected_result);
    delete_column(vals);
    delete_column(result);
    delete_column(ids);
}

TEST(TestStraightForwardImp,TestGreater ){
    size_t res_size =150 ;
    size_t input_size =150 ;
    auto ids = create_column(input_size,false);
    auto vals = create_column(input_size,false);
    auto result = create_column(res_size,true);
    size_t pred_val = 5 ;
    auto expected_result = generate_expected_result(input_size, [pred_val] (size_t val) -> size_t {
        return ! (val >pred_val);
    });
    auto predicate=   mondrian::vpipes::predicates::batched_predicates<size_t>::greater_than::straightforward_impl(pred_val);

    predicate(result, &res_size,
              ids, vals, input_size);

    EXPECT_EQ(has_same_vals(result,expected_result,res_size),true );
    delete_column(expected_result);
    delete_column(vals);
    delete_column(result);
    delete_column(ids);
}

TEST(TestStraightForwardImp,TestLessEqaul ){
    size_t res_size =150 ;
    size_t input_size =150 ;
    auto ids = create_column(input_size,false);
    auto vals = create_column(input_size,false);
    auto result = create_column(res_size,true);
    size_t pred_val = 5 ;
    auto expected_result = generate_expected_result(input_size, [pred_val] (size_t val) -> size_t {
        return ! (val <=pred_val);
    });
    auto predicate=   mondrian::vpipes::predicates::batched_predicates<size_t>::less_equal::straightforward_impl(pred_val);

    predicate(result, &res_size,
              ids, vals, input_size);

    EXPECT_EQ(has_same_vals(result,expected_result,res_size),true );
    delete_column(expected_result);
    delete_column(vals);
    delete_column(result);
    delete_column(ids);

}

TEST(TestStraightForwardImp,TestLess ){
    size_t res_size =150 ;
    size_t input_size =150 ;
    auto ids = create_column(input_size,false);
    auto vals = create_column(input_size,false);
    auto result = create_column(res_size,true);
    size_t pred_val = 5 ;
    auto expected_result = generate_expected_result(input_size, [pred_val] (size_t val) -> size_t {
        return ! (val <pred_val);
    });
    auto predicate=   mondrian::vpipes::predicates::batched_predicates<size_t>::less_than::straightforward_impl(pred_val);

    predicate(result, &res_size,
              ids, vals, input_size);

    EXPECT_EQ(has_same_vals(result,expected_result,res_size),true );
    delete_column(expected_result);
    delete_column(vals);
    delete_column(result);
    delete_column(ids);

}

TEST(TestStraightForwardImp,TestEqaul ){

    size_t res_size =150 ;
    size_t input_size =150 ;
    auto ids = create_column(input_size,false);
    auto vals = create_column(input_size,false);
    auto result = create_column(res_size,true);
    size_t pred_val = 5 ;
    auto expected_result = generate_expected_result(input_size, [pred_val] (size_t val) -> size_t {
        return ! (val ==pred_val);
    });
    auto predicate=   mondrian::vpipes::predicates::batched_predicates<size_t>::equal_to::straightforward_impl(pred_val);

    predicate(result, &res_size,
              ids, vals, input_size);

    EXPECT_EQ(has_same_vals(result,expected_result,res_size),true );
    delete_column(expected_result);
    delete_column(vals);
    delete_column(result);
    delete_column(ids);


}

TEST(TestStraightForwardImp,TestNotEqaul ){


    size_t res_size =150 ;
    size_t input_size =150 ;
    auto ids = create_column(input_size,false);
    auto vals = create_column(input_size,false);
    auto result = create_column(res_size,true);
    size_t pred_val = 5 ;
    auto expected_result = generate_expected_result(input_size, [pred_val] (size_t val) -> size_t {
        return ! (val !=pred_val);
    });
    auto predicate=   mondrian::vpipes::predicates::batched_predicates<size_t>::unequal_to::straightforward_impl(pred_val);

    predicate(result, &res_size,
              ids, vals, input_size);

    EXPECT_EQ(has_same_vals(result,expected_result,res_size),true );
    delete_column(expected_result);
    delete_column(vals);
    delete_column(result);
    delete_column(ids);


}



TEST(TestMicroOptimiziedImp,TestGreaterEqual ){

    size_t res_size =150 ;
    size_t input_size =150 ;
    auto ids = create_column(input_size,false);
    auto vals = create_column(input_size,false);
    auto result = create_column(res_size,true);
    size_t pred_val = 5 ;
    auto expected_result = generate_expected_result(input_size, [pred_val] (size_t val) -> size_t {
        return ! (val >=pred_val);
    });
    auto predicate=   mondrian::vpipes::predicates::batched_predicates<size_t>::greater_equal::micro_optimized_impl(pred_val,true);

    predicate(result, &res_size,
              ids, vals, input_size);

    EXPECT_EQ(has_same_vals(result,expected_result,res_size),true );
    delete_column(expected_result);
    delete_column(vals);
    delete_column(result);
    delete_column(ids);

}

TEST(TestMicroOptimiziedImp,TestGreater ){
    size_t res_size =150 ;
    size_t input_size =150 ;
    auto ids = create_column(input_size,false);
    auto vals = create_column(input_size,false);
    auto result = create_column(res_size,true);
    size_t pred_val = 5 ;
    auto expected_result = generate_expected_result(input_size, [pred_val] (size_t val) -> size_t {
        return ! (val >pred_val);
    });
    auto predicate=   mondrian::vpipes::predicates::batched_predicates<size_t>::greater_than::micro_optimized_impl(pred_val,true);

    predicate(result, &res_size,
              ids, vals, input_size);

    EXPECT_EQ(has_same_vals(result,expected_result,res_size),true );
    delete_column(expected_result);
    delete_column(vals);
    delete_column(result);
    delete_column(ids);

}

TEST(TestMicroOptimiziedImp,TestLessEqaul ){
    size_t res_size =150 ;
    size_t input_size =150 ;
    auto ids = create_column(input_size,false);
    auto vals = create_column(input_size,false);
    auto result = create_column(res_size,true);
    size_t pred_val = 5 ;
    auto expected_result = generate_expected_result(input_size, [pred_val] (size_t val) -> size_t {
        return ! (val <= pred_val);
    });
    auto predicate=   mondrian::vpipes::predicates::batched_predicates<size_t>::less_equal::micro_optimized_impl(pred_val,true);

    predicate(result, &res_size,
              ids, vals, input_size);

    EXPECT_EQ(has_same_vals(result,expected_result,res_size),true );
    delete_column(expected_result);
    delete_column(vals);
    delete_column(result);
    delete_column(ids);

}

TEST(TestMicroOptimiziedImp,TestLess ){
    size_t res_size =150 ;
    size_t input_size =150 ;
    auto ids = create_column(input_size,false);
    auto vals = create_column(input_size,false);
    auto result = create_column(res_size,true);
    size_t pred_val = 5 ;
    auto expected_result = generate_expected_result(input_size, [pred_val] (size_t val) -> size_t {
        return ! (val < pred_val);
    });
    auto predicate=   mondrian::vpipes::predicates::batched_predicates<size_t>::less_than::micro_optimized_impl(pred_val,true);

    predicate(result, &res_size,
              ids, vals, input_size);

    EXPECT_EQ(has_same_vals(result,expected_result,res_size),true );
    delete_column(expected_result);
    delete_column(vals);
    delete_column(result);
    delete_column(ids);

}

TEST(TestMicroOptimiziedImp,TestEqaul ){
    size_t res_size =150 ;
    size_t input_size =150 ;
    auto ids = create_column(input_size,false);
    auto vals = create_column(input_size,false);
    auto result = create_column(res_size,true);
    size_t pred_val = 5 ;
    auto expected_result = generate_expected_result(input_size, [pred_val] (size_t val) -> size_t {
        return ! (val == pred_val);
    });
    auto predicate=   mondrian::vpipes::predicates::batched_predicates<size_t>::equal_to::micro_optimized_impl(pred_val,true);

    predicate(result, &res_size,
              ids, vals, input_size);

    EXPECT_EQ(has_same_vals(result,expected_result,res_size),true );
    delete_column(expected_result);
    delete_column(vals);
    delete_column(result);
    delete_column(ids);

}

TEST(TestMicroOptimiziedImp,TestNotEqaul ){
    size_t res_size =150 ;
    size_t input_size =150 ;
    auto ids = create_column(input_size,false);
    auto vals = create_column(input_size,false);
    auto result = create_column(res_size,true);
    size_t pred_val = 5 ;
    auto expected_result = generate_expected_result(input_size, [pred_val] (size_t val) -> size_t {
        return ! (val != pred_val);
    });
    auto predicate=   mondrian::vpipes::predicates::batched_predicates<size_t>::unequal_to::micro_optimized_impl(pred_val,true);

    predicate(result, &res_size,
              ids, vals, input_size);

    EXPECT_EQ(has_same_vals(result,expected_result,res_size),true );
    delete_column(expected_result);
    delete_column(vals);
    delete_column(ids);
    delete_column(result);


}




TEST(TestBranchFreeImp,TestGreaterEqual ){
    size_t res_size =150 ;
    size_t input_size =150 ;
    auto ids = create_column(input_size,false);
    auto vals = create_column(input_size,false);
    auto result = create_column(res_size,true);
    size_t pred_val = 5 ;
    auto expected_result = generate_expected_result(input_size, [pred_val] (size_t val) -> size_t {
        return ! (val >=pred_val);
    });
    auto predicate=   mondrian::vpipes::predicates::batched_predicates<size_t>::greater_equal::branch_free_impl(pred_val,true);

    predicate(result, &res_size,
              ids, vals, input_size);

    EXPECT_EQ(has_same_vals(result,expected_result,res_size),true );
    delete_column(expected_result);
    delete_column(vals);
    delete_column(result);
    delete_column(ids);

}

TEST(TestBranchFreeImp,TestGreater ){
    size_t res_size =150 ;
    size_t input_size =150 ;
    auto ids = create_column(input_size,false);
    auto vals = create_column(input_size,false);
    auto result = create_column(res_size,true);
    size_t pred_val = 5 ;
    auto expected_result = generate_expected_result(input_size, [pred_val] (size_t val) -> size_t {
        return ! (val >pred_val);
    });
    auto predicate=   mondrian::vpipes::predicates::batched_predicates<size_t>::greater_than::branch_free_impl(pred_val,true);

    predicate(result, &res_size,
              ids, vals, input_size);

    EXPECT_EQ(has_same_vals(result,expected_result,res_size),true );
    delete_column(expected_result);
    delete_column(vals);
    delete_column(result);
    delete_column(ids);

}

TEST(TestBranchFreeImp,TestLessEqaul ){
    size_t res_size =150 ;
    size_t input_size =150 ;
    auto ids = create_column(input_size,false);
    auto vals = create_column(input_size,false);
    auto result = create_column(res_size,true);
    size_t pred_val = 5 ;
    auto expected_result = generate_expected_result(input_size, [pred_val] (size_t val) -> size_t {
        return ! (val <=pred_val);
    });
    auto predicate=   mondrian::vpipes::predicates::batched_predicates<size_t>::less_equal::branch_free_impl(pred_val,true);

    predicate(result, &res_size,
              ids, vals, input_size);
    EXPECT_EQ(has_same_vals(result,expected_result,res_size),true );
    delete_column(expected_result);
    delete_column(vals);
    delete_column(result);
    delete_column(ids);

}

TEST(TestBranchFreeImp,TestLess ){
    size_t res_size =150 ;
    size_t input_size =150 ;
    auto vals = create_column(input_size,false);
    auto ids = create_column(input_size,false);
    auto result = create_column(res_size,true);
    size_t pred_val = 5 ;
    auto expected_result = generate_expected_result(input_size, [pred_val] (size_t val) -> size_t {
        return ! (val <pred_val);
    });
    auto predicate=   mondrian::vpipes::predicates::batched_predicates<size_t>::less_than::branch_free_impl(pred_val,true);

    predicate(result, &res_size,
              ids, vals, input_size);

    EXPECT_EQ(has_same_vals(result,expected_result,res_size),true );
    delete_column(expected_result);
    delete_column(vals);
    delete_column(result);
    delete_column(ids);

}

TEST(TestBranchFreeImp,TestEqaul ){
    size_t res_size =150 ;
    size_t input_size =150 ;
    auto ids = create_column(input_size,false);
    auto vals = create_column(input_size,false);
    auto result = create_column(res_size,true);
    size_t pred_val = 5 ;
    auto expected_result = generate_expected_result(input_size, [pred_val] (size_t val) -> size_t {
        return ! (val ==pred_val);
    });
    auto predicate=   mondrian::vpipes::predicates::batched_predicates<size_t>::equal_to::branch_free_impl(pred_val,true);

    predicate(result, &res_size,
              ids, vals, input_size);

    EXPECT_EQ(has_same_vals(result,expected_result,res_size),true );
    delete_column(expected_result);
    delete_column(vals);
    delete_column(result);
    delete_column(ids);

}

TEST(TestBranchFreeImp,TestNotEqaul ){
    size_t res_size =150 ;
    size_t input_size =150 ;
    auto ids = create_column(input_size,false);
    auto vals = create_column(input_size,false);
    auto result = create_column(res_size,true);
    size_t pred_val = 5 ;
    auto expected_result = generate_expected_result(input_size, [pred_val] (size_t val) -> size_t {
        return ! (val !=pred_val);
    });
    auto predicate=   mondrian::vpipes::predicates::batched_predicates<size_t>::unequal_to::branch_free_impl(pred_val,true);

    predicate(result, &res_size,
              ids, vals, input_size);

    EXPECT_EQ(has_same_vals(result,expected_result,res_size),true );
    delete_column(expected_result);
    delete_column(vals);
    delete_column(result);
    delete_column(ids);

}