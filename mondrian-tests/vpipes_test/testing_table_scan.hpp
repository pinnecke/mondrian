//
// Created by Mahmoud Mohsen on 3/17/17.
//
#pragma once
#include "gtest/gtest.h"
#include <vpipes.hpp>
#include <testing_utilities.hpp>


TEST(TestTableScan,TestOneInterval){
    size_t chunk_size =5;
    size_t result_size =1000;
    auto  result = create_column<size_t >(result_size);
    mondrian::vpipes::functional::batched_materializes<size_t>::func_t materialize1 = [] (size_t *out_begin, size_t *out_end,
                                                                                          const size_t *begin, const size_t*end)
    {
        assert (out_end - out_begin >= end - begin);
        size_t distance = (end - begin);
        for (size_t i = 0; i != distance; ++i) {
            *(out_begin+i) = *(begin+i);
        }
    };
    toolkit::materialize<size_t > mat (result,&result_size,materialize1,chunk_size);
    auto num_intervals = 1 ;
    interval<size_t > *intervals  = new  interval<size_t > [num_intervals] ;
    intervals [0] =interval <size_t > (50 ,80);
    size_t *starts = new size_t [num_intervals];
    starts [0] =50 ;
    size_t *ends  = new size_t [num_intervals];
    ends[0] = 80 ;
    vector<size_t > my_vec = generate_vector_from_intervals(starts ,ends,num_intervals,2);
    delete []starts;
    delete []ends;
    toolkit::table_scan<size_t > my_table (&mat, intervals ,  intervals +num_intervals ,PREDICATE(2),materialize1,chunk_size);
    my_table.start();
    EXPECT_EQ(  has_same_vals(my_vec.data(),result,result_size),1);
    delete_column<size_t >(result);
    delete [] intervals;
}


TEST(TestTableScan,TestManyIntervals){
    size_t chunk_size =5;
    size_t result_size =1000;
    auto  result = create_column<size_t >(result_size);
    mondrian::vpipes::functional::batched_materializes<size_t>::func_t materialize1 = [] (size_t *out_begin, size_t *out_end,
                                                                                          const size_t *begin, const size_t*end)
    {
        assert (out_end - out_begin >= end - begin);
        size_t distance = (end - begin);
        for (size_t i = 0; i != distance; ++i) {
            *(out_begin+i) = *(begin+i);
        }
    };
    toolkit::materialize<size_t > mat (result,&result_size,materialize1,chunk_size);
    auto num_intervals = 10  ;
    interval<size_t > *intervals  = new  interval<size_t > [num_intervals] ;
    size_t *starts = new size_t [num_intervals];
    size_t *ends  = new size_t [num_intervals];
     auto offset =10 ;
    for (auto i = 0 ; i <num_intervals ; i++){
                starts[i]=offset;
                ends[i] = starts[i]+20;
                intervals [i] =interval <size_t > (starts[i] ,ends[i]);
                offset = ends[i] +10 ;
    }

    vector<size_t > my_vec = generate_vector_from_intervals(starts ,ends,num_intervals,2);
    delete []starts;
    delete []ends;
    toolkit::table_scan<size_t > my_table (&mat, intervals ,  intervals +num_intervals ,PREDICATE(2),materialize1,chunk_size);
    my_table.start();
    EXPECT_EQ(  has_same_vals(my_vec.data(),result,result_size),1);
    delete_column<size_t >(result);
    delete [] intervals;
}


TEST(TestTableScan,TestOverlappingIntervals){
    size_t chunk_size =5;
    size_t result_size =1000;
    auto  result = create_column<size_t >(result_size);
    mondrian::vpipes::functional::batched_materializes<size_t>::func_t materialize1 = [] (size_t *out_begin, size_t *out_end,
                                                                                          const size_t *begin, const size_t*end)
    {
        assert (out_end - out_begin >= end - begin);
        size_t distance = (end - begin);
        for (size_t i = 0; i != distance; ++i) {
            *(out_begin+i) = *(begin+i);
        }
    };
    toolkit::materialize<size_t > mat (result,&result_size,materialize1,chunk_size);
    auto num_intervals = 2  ;
    interval<size_t > *intervals  = new  interval<size_t > [num_intervals] ;
    size_t *starts = new size_t [num_intervals];
    size_t *ends  = new size_t [num_intervals];
    starts[0] = 30 ; ends [0] = 80;
    intervals[0] = interval<size_t> (starts[0],ends[0]);
    starts[1] =40; ends [1] =100;
    intervals[1] = interval<size_t> (starts[1],ends[1]);
    vector<size_t > my_vec = generate_vector_from_intervals(starts ,ends,num_intervals,2);
    delete []starts;
    delete []ends;
    toolkit::table_scan<size_t > my_table (&mat, intervals ,  intervals +num_intervals ,PREDICATE(2),materialize1,chunk_size);
    my_table.start();

    EXPECT_EQ(  true,false)<<"it is not allowed to have interleaved intervals so the programm should not reach this part.";
    delete_column<size_t >(result);
    delete [] intervals;
}
