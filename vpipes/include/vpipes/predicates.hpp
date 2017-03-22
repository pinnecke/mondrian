// Vector-Pipes - a framework for the push-based iterator model with support of vectorized execution
// Copyright (C) 2017  Marcus Pinnecke (marcus.pinnecke@ovgu.de)
//
// This program is free software; you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
// even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License al ong with this program; if
// not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
// Boston, MA  02110-1301, USA.

#pragma once

#include <vpipes.hpp>

#define ASSERT_VALID_BATCHED_PREDICATE_ARGS()                           \
{                                                                       \
    assert (*result_size == 0);                                         \
    assert (result_buffer != nullptr);                                  \
    assert (values_begin != nullptr && values_end != nullptr);          \
    assert (values_begin <= values_end);                                \
    assert (tupletids_begin != nullptr && tupletids_end != nullptr);    \
    assert (tupletids_begin <= tupletids_end);                          \
}

#define ASSERT_VALID_BATCHED_PREDICATE_ARGS2()                          \
{                                                                       \
    assert (out_matching_indices != nullptr);                           \
    assert (out_num_matching_indices != nullptr);                       \
    assert (tupletids != nullptr);                                      \
    assert (values != nullptr);                                         \
}

#define POINTER_DISTANCE(begin, end)                                    \
    (end - begin)

#define DEFINE_STRAIGHT_FORWARD(name, opp)                                                                      \
struct name                                                                                                     \
{                                                                                                               \
    value_t compare_value;                                                                                      \
                                                                                                                \
    explicit name(value_t compare_value): compare_value(compare_value)  { }                                     \
                                                                                                                \
    virtual inline void operator()(size_t *out_matching_indices, size_t *out_num_matching_indices,              \
                           const tupletid_t *tupletids, const value_t *values,                                  \
                           size_t num_elements) final __attribute__((always_inline))                            \
    {                                                                                                           \
        ASSERT_VALID_BATCHED_PREDICATE_ARGS2();                                                                 \
        const size_t *out_matching_indices_start = out_matching_indices;                                        \
        for (size_t idx = 0; idx != num_elements; ++idx) {                                                      \
            if (values[idx] opp compare_value) {                                                                \
                  *out_matching_indices++ = idx;                                                                \
            }                                                                                                   \
        }                                                                                                       \
        *out_num_matching_indices = (out_matching_indices - out_matching_indices_start);                        \
    }                                                                                                           \
};

#define DEFINE_MICRO_OPTIMIZED_PREDICATE(name, opp)                                                             \
struct name                                                                                                     \
{                                                                                                               \
    value_t compare_value;                                                                                      \
    bool hint_expected_true;                                                                                    \
                                                                                                                \
    explicit name(value_t compare_value, bool hint_expected_true): compare_value(compare_value),                \
                                                                    hint_expected_true(hint_expected_true) { }  \
                                                                                                                \
    virtual inline void operator()(size_t *out_matching_indices, size_t *out_num_matching_indices,              \
                           const tupletid_t *tupletids, const value_t *values,                                  \
                           size_t num_elements) final __attribute__((always_inline))                            \
    {                                                                                                           \
        ASSERT_VALID_BATCHED_PREDICATE_ARGS2();                                                                 \
        const size_t *out_matching_indices_start = out_matching_indices;                                        \
        __builtin_prefetch(out_matching_indices, PREFETCH_RW_FOR_WRITE, PREFETCH_LOCALITY_REMOVE_FROM_CACHE);   \
        __builtin_prefetch(out_num_matching_indices, PREFETCH_RW_FOR_WRITE, PREFETCH_LOCALITY_REMOVE_FROM_CACHE);   \
        __builtin_prefetch(tupletids, PREFETCH_RW_FOR_READ, PREFETCH_LOCALITY_REMOVE_FROM_CACHE);               \
        __builtin_prefetch(values, PREFETCH_RW_FOR_READ, PREFETCH_LOCALITY_REMOVE_FROM_CACHE);                  \
        for (size_t idx = 0; idx != num_elements; ++idx) {                                                      \
            if (__builtin_expect((values[idx] opp compare_value), hint_expected_true)) {                        \
                  *out_matching_indices++ = idx;                                                                \
            }                                                                                                   \
        }                                                                                                       \
        *out_num_matching_indices = (out_matching_indices - out_matching_indices_start);                        \
    }                                                                                                           \
};

namespace mondrian
{
    namespace vpipes
    {
        namespace predicates
        {
            template<class ValueType, class TupletIdType = size_t>
            struct batched_predicates
            {
                using value_t = ValueType;
                using tupletid_t = TupletIdType;
                using func_t = std::function<void(size_t *out_matching_indices, size_t *out_num_matching_indices,
                                                  const tupletid_t *tupletids, const value_t *values,
                                                  size_t num_elements)>;

                struct less_than
                {
                    DEFINE_STRAIGHT_FORWARD(straightforward_impl, <);
                    DEFINE_MICRO_OPTIMIZED_PREDICATE(micro_optimized_impl, <);
                };

                struct less_equal
                {
                    DEFINE_STRAIGHT_FORWARD(straightforward_impl, <=);
                    DEFINE_MICRO_OPTIMIZED_PREDICATE(micro_optimized_impl, <=);
                };

                struct equal_to
                {
                    DEFINE_STRAIGHT_FORWARD(straightforward_impl, ==);
                    DEFINE_MICRO_OPTIMIZED_PREDICATE(micro_optimized_impl, ==);
                };

                struct unequal_to
                {
                    DEFINE_STRAIGHT_FORWARD(straightforward_impl, !=);
                    DEFINE_MICRO_OPTIMIZED_PREDICATE(micro_optimized_impl, !=);
                };

                struct greater_equal
                {
                    DEFINE_STRAIGHT_FORWARD(straightforward_impl, >=);
                    DEFINE_MICRO_OPTIMIZED_PREDICATE(micro_optimized_impl, >=);
                };

                struct greater_than
                {
                    DEFINE_STRAIGHT_FORWARD(straightforward_impl, >);
                    DEFINE_MICRO_OPTIMIZED_PREDICATE(micro_optimized_impl, >);
                };
            };
        }
    }
}

