//
// Created by Mahmoud Mohsen on 3/16/17.
//

#pragma once

#include "gtest/gtest.h"
#include <vpipes.hpp>
#include <testing_utilities.hpp>

using namespace mondrian::vpipes;


TEST(TESTfilters, BasicFiltersTest) {
    size_t num_elements = 100;
    auto input = create_column<size_t>(num_elements, false, true);
    size_t chunk_size = 20;
    auto result_size = 100;
    auto result = create_column<size_t>(num_elements);
    auto input_filtered = create_column<size_t>(result_size, false, true);
    auto index = 0;
    for (auto i = input; i != (input + num_elements); ++i) {
        if (((*i) % 2 == 0))
            input_filtered[index++] = *i;

    }
    mondrian::vpipes::functional::batched_materializes<size_t>::func_t materialize1 = [](size_t *out_begin,
                                                                                         size_t *out_end,
                                                                                         const size_t *begin,
                                                                                         const size_t *end) {
        assert (out_end - out_begin >= end - begin);
        size_t distance = (end - begin);
        for (size_t i = 0; i != distance; ++i) {
            *(out_begin + i) = *(begin + i);
        }
    };

    toolkit::materialize<size_t> mat(result, &num_elements, materialize1, chunk_size);
    toolkit::filter<size_t> even_num_filter(&mat, materialize1, PREDICATE(2), chunk_size);
    auto read = toolkit::reader<std::size_t>(&even_num_filter, input, input + num_elements, chunk_size);
    read.start();
    EXPECT_EQ(has_same_vals(input_filtered, result, num_elements), true);
    delete_column<size_t>(input_filtered);
    delete_column<size_t>(result);
    delete_column<size_t>(input);

}


TEST(TESTfilters, CascadingFilters) {
    size_t num_elements = 100;
    auto input = create_column<size_t>(num_elements, false, true);
    size_t chunk_size = 20;
    auto result_size = 100;
    auto result = create_column<size_t>(num_elements);
    auto input_filtered = create_column<size_t>(result_size, false, true);
    auto index = 0;
    for (auto i = input; i != (input + num_elements); ++i) {
        if (((*i) % 2 == 0) && ((*i) % 5 == 0) && ((*i) % 12 == 0))
            input_filtered[index++] = *i;

    }

    mondrian::vpipes::functional::batched_materializes<size_t>::func_t materialize1 = [](size_t *out_begin,
                                                                                         size_t *out_end,
                                                                                         const size_t *begin,
                                                                                         const size_t *end) {
        assert (out_end - out_begin >= end - begin);
        size_t distance = (end - begin);
        for (size_t i = 0; i != distance; ++i) {
            *(out_begin + i) = *(begin + i);
        }
    };

    toolkit::materialize<size_t> mat(result, &num_elements, materialize1, chunk_size);
    toolkit::filter<size_t> fives_filter(&mat, materialize1, PREDICATE(5), chunk_size);
    toolkit::filter<size_t> even_num_filter(&fives_filter, materialize1, PREDICATE(2), chunk_size);
    toolkit::filter<size_t> twelve_filter(&even_num_filter, materialize1, PREDICATE(12), chunk_size);
    auto read = toolkit::reader<std::size_t>(&twelve_filter, input, input + num_elements, chunk_size);
    read.start();
    EXPECT_EQ(has_same_vals(input_filtered, result, num_elements), true);
    delete_column<size_t>(input_filtered);
    delete_column<size_t>(result);
    delete_column<size_t>(input);

}


TEST(TESTfilters, NoConditionSatisfied) {
    size_t num_elements = 100;
    auto input = create_column<size_t>(num_elements, false, true);
    size_t chunk_size = 20;
    auto result_size = 100;
    auto result = create_column<size_t>(num_elements);
    auto input_filtered = create_column<size_t>(result_size, false, true);
    auto index = 0;
    for (auto i = input; i != (input + num_elements); ++i) {
        if (((*i) % 1000000 == 0))
            input_filtered[index++] = *i;

    }
    mondrian::vpipes::functional::batched_materializes<size_t>::func_t materialize1 = [](size_t *out_begin,
                                                                                         size_t *out_end,
                                                                                         const size_t *begin,
                                                                                         const size_t *end) {
        assert (out_end - out_begin >= end - begin);
        size_t distance = (end - begin);
        for (size_t i = 0; i != distance; ++i) {
            *(out_begin + i) = *(begin + i);
        }
    };

    toolkit::materialize<size_t> mat(result, &num_elements, materialize1, chunk_size);
    toolkit::filter<size_t> dummy_filter(&mat, materialize1, PREDICATE(1000000), chunk_size);
    auto read = toolkit::reader<std::size_t>(&dummy_filter, input, input + num_elements, chunk_size);
    read.start();
    EXPECT_EQ(has_same_vals(input_filtered, result, num_elements), true);
    delete_column<size_t>(input_filtered);
    delete_column<size_t>(result);
    delete_column<size_t>(input);

}