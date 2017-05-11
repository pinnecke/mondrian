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

#define ROUND_DOWN(x, s) ((x) & ~((s)-1))

#define ASSERT_VALID_BATCHED_PREDICATE_ARGS()                           \
{                                                                       \
    assert (out_matching_indices != nullptr);                           \
    assert (out_num_matching_indices != nullptr);                       \
    assert (statistics != nullptr);                                     \
    assert (tupletids != nullptr);                                      \
    assert (values != nullptr);                                         \
    assert (null_mask != nullptr);                                      \
}

#define POINTER_DISTANCE(begin, end)                                    \
    (end - begin)

#define DEFINE_STRAIGHT_FORWARD(name, opp)                                                                             \
struct name                                                                                                            \
{                                                                                                                      \
    value_t compare_value;                                                                                             \
                                                                                                                       \
    explicit name(value_t compare_value): compare_value(compare_value)  { }                                            \
                                                                                                                       \
    virtual inline void operator()(__out__ size_t *out_matching_indices,                                               \
                                   __out__ size_t *out_num_matching_indices,                                           \
                                   __out__ statistics::predicate_run *statistics,                                      \
                                   __in__ null_value_filter_policy null_policy,                                        \
                                   __in__ const tuplet_id_t *tupletids,                                                \
                                   __in__ const value_t *values,                                                       \
                                   __in__ const mtl::smart_bitmask *null_mask,                                         \
                                   __in__ size_t num_elements) final __attribute__((always_inline))                    \
    {                                                                                                                  \
        ASSERT_VALID_BATCHED_PREDICATE_ARGS();                                                                         \
        const size_t *out_matching_indices_start = out_matching_indices;                                               \
        if (null_mask->is_unset()) {																				   \
            statistics->count_non_null_branch_used++;                                                                  \
            for (size_t idx = 0; idx != num_elements; ++idx) {                                                         \
                if (values[idx] opp compare_value) {                                                                   \
                      *out_matching_indices++ = idx;                                                                   \
                      statistics->count_satisfying_values++;                                                           \
                } else {                                                                                               \
                      statistics->count_non_satisfying_values++;                                                       \
                }                                                                                                      \
            }                                                                                                          \
        } else {                                                                                                       \
            bool tuplet_is_null;                                                                                       \
            statistics->count_null_branch_used++;                                                                      \
            for (size_t idx = 0; idx != num_elements; ++idx) {                                                         \
                SMART_BITMASK_GET_UNSAFE_FAST(tuplet_is_null, null_mask, idx);                                         \
                if (tuplet_is_null) {                                                                                  \
                    if (null_policy == null_value_filter_policy::skip_null_values) {                                   \
                        *out_matching_indices++ = idx;                                                                 \
                        statistics->count_skipped_values++;                                                            \
                    }                                                                                                  \
                    statistics->count_null_values++;                                                                   \
                } else {                                                                                               \
                    if (values[idx] opp compare_value) { 		                                                       \
                        *out_matching_indices++ = idx;                                                                 \
                        statistics->count_satisfying_values++;                                                         \
                    } else {                                                                                           \
                        statistics->count_non_satisfying_values++;                                                     \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        *out_num_matching_indices = (out_matching_indices - out_matching_indices_start);                               \
    }                                                                                                                  \
};

#define DEFINE_MICRO_OPTIMIZED_PREDICATE(name, opp)                                                                    \
struct name                                                                                                            \
{                                                                                                                      \
    value_t compare_value;                                                                                             \
    bool hint_expected_true;                                                                                           \
                                                                                                                       \
    explicit name(__in__ value_t compare_value,                                                                        \
                  __in__ bool hint_expected_true): compare_value(compare_value),                                       \
                                                   hint_expected_true(hint_expected_true) { }                          \
                                                                                                                       \
    virtual inline void operator()(__out__ size_t *out_matching_indices,                                               \
                                   __out__ size_t *out_num_matching_indices,                                           \
                                   __out__ statistics::predicate_run *statistics,                                      \
                                   __in__ null_value_filter_policy null_policy,                                        \
                                   __in__ const tuplet_id_t *tupletids,                                                \
                                   __in__ const value_t *values,                                                       \
                                   __in__ const mtl::smart_bitmask *null_mask,                                         \
                           size_t num_elements) final __attribute__((always_inline))                                   \
    {                                                                                                                  \
        ASSERT_VALID_BATCHED_PREDICATE_ARGS();                                                                         \
        const size_t *out_matching_indices_start = out_matching_indices;                                               \
        __builtin_prefetch(out_matching_indices, PREFETCH_RW_FOR_WRITE, PREFETCH_LOCALITY_REMOVE_FROM_CACHE);          \
        __builtin_prefetch(out_num_matching_indices, PREFETCH_RW_FOR_WRITE, PREFETCH_LOCALITY_REMOVE_FROM_CACHE);      \
        __builtin_prefetch(tupletids, PREFETCH_RW_FOR_READ, PREFETCH_LOCALITY_REMOVE_FROM_CACHE);                      \
        __builtin_prefetch(values, PREFETCH_RW_FOR_READ, PREFETCH_LOCALITY_REMOVE_FROM_CACHE);                         \
        if (null_mask->is_unset()) {																				   \
            statistics->count_non_null_branch_used++;                                                                  \
	        for (size_t idx = 0; idx != num_elements; ++idx) {                                                         \
                if (__builtin_expect((values[idx] opp compare_value), hint_expected_true)){                            \
                    *out_matching_indices++ = idx;                                                                     \
                    statistics->count_satisfying_values++;                                                             \
                } else {                                                                                               \
                    statistics->count_non_satisfying_values++;                                                         \
                }                                                                                                      \
            }																										   \
        } else {                                                                                                       \
            for (size_t idx = 0; idx != num_elements; ++idx) {                                                         \
                bool is_non_null;                                                                                      \
                statistics->count_null_branch_used++;                                                                  \
                SMART_BITMASK_GET_UNSAFE_FAST(is_non_null, null_mask, idx);                                            \
                if (is_non_null) {                                                                                     \
                    if (__builtin_expect((values[idx] opp compare_value), hint_expected_true)) { 		               \
                        *out_matching_indices++ = idx;                                                                 \
                        statistics->count_satisfying_values++;                                                         \
                    } else {                                                                                           \
                        statistics->count_non_satisfying_values++;                                                     \
                    }                                                                                                  \
                } else {                                                                                               \
                    if (null_policy == null_value_filter_policy::skip_null_values) {                                   \
                        *out_matching_indices++ = idx;                                                                 \
                        statistics->count_skipped_values++;                                                            \
                    }                                                                                                  \
                    statistics->count_null_values++;                                                                   \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        *out_num_matching_indices = (out_matching_indices - out_matching_indices_start);                               \
    }                                                                                                                  \
};


#define BRANCH_FREE_PREDICATE(name, opp)                                                                               \
struct name                                                                                                            \
{                                                                                                                      \
    value_t compare_value;                                                                                             \
                                                                                                                       \
    explicit name(value_t compare_value): compare_value(compare_value)  { }                                            \
                                                                                                                       \
    virtual inline void operator()(__out__ size_t *out_matching_indices,                                               \
                                   __out__ size_t *out_num_matching_indices,                                           \
                                   __out__ statistics::predicate_run *statistics,                                      \
                                   __in__ null_value_filter_policy null_policy,                                        \
                                   __in__ const tuplet_id_t *tupletids,                                                \
                                   __in__ const value_t *values,                                                       \
                                   __in__ const mtl::smart_bitmask *null_mask,                                         \
                                   __in__ size_t num_elements) final __attribute__((always_inline))                    \
    {                                                                                                                  \
        ASSERT_VALID_BATCHED_PREDICATE_ARGS();                                                                         \
        const size_t *out_matching_indices_start = out_matching_indices;                                               \
        if (null_mask->is_unset()) {																				   \
            statistics->count_non_null_branch_used++;                                                                  \
            size_t idx = 0;                                                                                            \
            for  ( ; num_elements - idx; ++idx) {                                                                      \
            *out_matching_indices = idx;                                                                               \
             out_matching_indices = out_matching_indices+((values[idx] opp compare_value));                            \
             statistics->count_satisfying_values = statistics->count_satisfying_values +                               \
                                                   (values[idx] opp compare_value);                                    \
             statistics->count_non_satisfying_values = statistics->count_non_satisfying_values +                       \
                                                       !(values[idx] opp compare_value);                               \
             }                                                                                                         \
        } else {                                                                                                       \
            bool tuplet_is_null;                                                                                       \
            bool skip_for_null= false;                                                                                 \
            statistics->count_null_branch_used++;                                                                      \
            size_t idx = 0;                                                                                            \
            for ( ; num_elements - idx; ++idx) {                                                                       \
                 SMART_BITMASK_GET_UNSAFE_FAST(tuplet_is_null, null_mask, idx);                                        \
                 *out_matching_indices = idx;                                                                          \
                 statistics->count_satisfying_values = statistics->count_satisfying_values +                           \
                                                   ((!tuplet_is_null && values[idx] opp compare_value));               \
                 statistics->count_null_values = statistics->count_null_values + tuplet_is_null;                       \
                 statistics->count_non_satisfying_values = statistics->count_non_satisfying_values +                   \
                                                           (!tuplet_is_null && !(values[idx] opp compare_value) );     \
                 skip_for_null = tuplet_is_null && null_policy == null_value_filter_policy::skip_null_values;          \
                 *out_matching_indices = ((*out_matching_indices) * (1 - skip_for_null)) + (idx * skip_for_null);      \
                 out_matching_indices = out_matching_indices +                                                         \
                                        ((!tuplet_is_null && values[idx] opp compare_value) || skip_for_null);         \
                 statistics->count_skipped_values = statistics->count_skipped_values + skip_for_null;                  \
                 }                                                                                                     \
            }                                                                                                          \
        *out_num_matching_indices = (out_matching_indices - out_matching_indices_start);                               \
    }                                                                                                                  \
};

#define MICRO_BRANCH_FREE_PREDICATE(name, opp)                                                                         \
struct name                                                                                                            \
{                                                                                                                      \
    value_t compare_value;                                                                                             \
    bool hint_expected_true;                                                                                           \
                                                                                                                       \
    explicit name(__in__ value_t compare_value,                                                                        \
                  __in__ bool hint_expected_true): compare_value(compare_value),                                       \
                                                   hint_expected_true(hint_expected_true) { }                          \
                                                                                                                       \
    virtual inline void operator()(__out__ size_t *out_matching_indices,                                               \
                                   __out__ size_t *out_num_matching_indices,                                           \
                                   __out__ statistics::predicate_run *statistics,                                      \
                                   __in__ null_value_filter_policy null_policy,                                        \
                                   __in__ const tuplet_id_t *tupletids,                                                \
                                   __in__ const value_t *values,                                                       \
                                   __in__ const mtl::smart_bitmask *null_mask,                                         \
                           size_t num_elements) final __attribute__((always_inline))                                   \
    {                                                                                                                  \
        ASSERT_VALID_BATCHED_PREDICATE_ARGS();                                                                         \
        const size_t *out_matching_indices_start = out_matching_indices;                                               \
        if (null_mask->is_unset()) {																				   \
            statistics->count_non_null_branch_used++;                                                                  \
            size_t idx = 0;                                                                                            \
            for ( ; num_elements - idx; ++idx) {                                                                       \
            *out_matching_indices = idx;                                                                               \
             out_matching_indices = out_matching_indices + (__builtin_expect((values[idx]                              \
                                    opp compare_value), hint_expected_true));                                          \
             statistics->count_satisfying_values = statistics->count_satisfying_values +                               \
                                                   (__builtin_expect((values[idx] opp compare_value),                  \
                                                   hint_expected_true));                                               \
             statistics->count_non_satisfying_values = statistics->count_non_satisfying_values +                       \
                                                       !(__builtin_expect((values[idx] opp compare_value),             \
                                                       hint_expected_true));                                           \
             }                                                                                                         \
        } else {                                                                                                       \
            bool tuplet_is_null;                                                                                       \
            bool skip_for_null= false;                                                                                 \
            statistics->count_null_branch_used++;                                                                      \
            size_t idx = 0;                                                                                            \
            for  ( ; num_elements - idx; ++idx) {                                                                      \
             SMART_BITMASK_GET_UNSAFE_FAST(tuplet_is_null, null_mask, idx);                                            \
             *out_matching_indices = idx;                                                                              \
             statistics->count_satisfying_values = statistics->count_satisfying_values + ((!tuplet_is_null &&          \
                                                   (__builtin_expect((values[idx] opp compare_value),                  \
                                                   hint_expected_true))));                                             \
             statistics->count_null_values = statistics->count_null_values + tuplet_is_null;                           \
             statistics->count_non_satisfying_values = statistics->count_non_satisfying_values +                       \
                                                       (!tuplet_is_null && !(__builtin_expect((values[idx]             \
                                                       opp compare_value), hint_expected_true)));                      \
             skip_for_null = tuplet_is_null && null_policy == null_value_filter_policy::skip_null_values;              \
             *out_matching_indices = ((*out_matching_indices) * (1 - skip_for_null)) + (idx * skip_for_null);          \
             out_matching_indices = out_matching_indices + ((!tuplet_is_null &&                                        \
                                    (__builtin_expect((values[idx] opp compare_value),                                 \
                                    hint_expected_true))) || skip_for_null);                                           \
             statistics->count_skipped_values = statistics->count_skipped_values + skip_for_null;                      \
                    }                                                                                                  \
                }                                                                                                      \
        *out_num_matching_indices = (out_matching_indices - out_matching_indices_start);                               \
    }                                                                                                                  \
};

#define MICRO_UNROLLED_BRANCH_FREE_PREDICATE(name, opp)                                                                \
struct name                                                                                                            \
{                                                                                                                      \
    value_t compare_value;                                                                                             \
    bool hint_expected_true;                                                                                           \
                                                                                                                       \
    explicit name(__in__ value_t compare_value,                                                                        \
                  __in__ bool hint_expected_true): compare_value(compare_value),                                       \
                                                   hint_expected_true(hint_expected_true) { }                          \
                                                                                                                       \
    virtual inline void operator()(__out__ size_t *out_matching_indices,                                               \
                                   __out__ size_t *out_num_matching_indices,                                           \
                                   __out__ statistics::predicate_run *statistics,                                      \
                                   __in__ null_value_filter_policy null_policy,                                        \
                                   __in__ const tuplet_id_t *tupletids,                                                \
                                   __in__ const value_t *values,                                                       \
                                   __in__ const mtl::smart_bitmask *null_mask,                                         \
                           size_t num_elements) final __attribute__((always_inline))                                   \
    {                                                                                                                  \
        ASSERT_VALID_BATCHED_PREDICATE_ARGS();                                                                         \
        const size_t *out_matching_indices_start = out_matching_indices;                                               \
        bool elements_bigger_1 = num_elements > 1;                                                                     \
        if (null_mask->is_unset()) {																				   \
            statistics->count_non_null_branch_used++;                                                                  \
            size_t idx = 0;                                                                                            \
            while (elements_bigger_1 && ROUND_DOWN(num_elements, 2) - idx) {                                           \
                *out_matching_indices = idx;                                                                           \
                out_matching_indices = out_matching_indices + (__builtin_expect((values[idx] opp compare_value),       \
                                       hint_expected_true));                                                           \
                statistics->count_satisfying_values = statistics->count_satisfying_values +                            \
                                                      (__builtin_expect((values[idx] opp compare_value),               \
                                                      hint_expected_true));                                            \
                statistics->count_non_satisfying_values = statistics->count_non_satisfying_values +                    \
                                                          !(__builtin_expect((values[idx] opp compare_value),          \
                                                          hint_expected_true));                                        \
                ++idx;                                                                                                 \
                *out_matching_indices = idx;                                                                           \
                out_matching_indices = out_matching_indices + (__builtin_expect((values[idx] opp compare_value),       \
                                       hint_expected_true));                                                           \
                statistics->count_satisfying_values = statistics->count_satisfying_values +                            \
                                                      (__builtin_expect((values[idx] opp compare_value),               \
                                                      hint_expected_true));                                            \
                statistics->count_non_satisfying_values = statistics->count_non_satisfying_values +                    \
                                                          !(__builtin_expect((values[idx] opp compare_value),          \
                                                          hint_expected_true));                                        \
                ++idx;                                                                                                 \
            }                                                                                                          \
            while (num_elements - idx) {                                                                               \
                 *out_matching_indices = idx;                                                                          \
                 out_matching_indices = out_matching_indices + (__builtin_expect((values[idx] opp compare_value),      \
                                        hint_expected_true));                                                          \
                 statistics->count_satisfying_values = statistics->count_satisfying_values +                           \
                                                       (__builtin_expect((values[idx] opp compare_value),              \
                                                       hint_expected_true));                                           \
                 statistics->count_non_satisfying_values = statistics->count_non_satisfying_values +                   \
                                                           !(__builtin_expect((values[idx] opp compare_value),         \
                                                           hint_expected_true));                                       \
                 ++idx;                                                                                                \
            }                                                                                                          \
        } else {                                                                                                       \
            bool tuplet_is_null;                                                                                       \
            bool skip_for_null= false;                                                                                 \
            statistics->count_null_branch_used++;                                                                      \
            size_t idx = 0;                                                                                            \
            while (elements_bigger_1 && ROUND_DOWN(num_elements, 2) - idx) {                                           \
                SMART_BITMASK_GET_UNSAFE_FAST(tuplet_is_null, null_mask, idx);                                         \
                *out_matching_indices = idx;                                                                           \
                statistics->count_satisfying_values = statistics->count_satisfying_values + ((!tuplet_is_null &&       \
                                                      (__builtin_expect((values[idx] opp compare_value),               \
                                                      hint_expected_true))));                                          \
                statistics->count_null_values = statistics->count_null_values + tuplet_is_null;                        \
                statistics->count_non_satisfying_values = statistics->count_non_satisfying_values +                    \
                                                          (!tuplet_is_null &&                                          \
                                                          !(__builtin_expect((values[idx] opp compare_value),          \
                                                          hint_expected_true)));                                       \
                skip_for_null = tuplet_is_null && null_policy == null_value_filter_policy::skip_null_values;           \
                *out_matching_indices = ((*out_matching_indices) * ( 1- skip_for_null)) + (idx * skip_for_null);       \
                out_matching_indices = out_matching_indices + ((!tuplet_is_null &&                                     \
                                       (__builtin_expect((values[idx] opp compare_value),                              \
                                       hint_expected_true))) ||                                                        \
                                       skip_for_null);                                                                 \
                statistics->count_skipped_values = statistics->count_skipped_values + skip_for_null;                   \
                ++idx;                                                                                                 \
                SMART_BITMASK_GET_UNSAFE_FAST(tuplet_is_null, null_mask, idx);                                         \
                *out_matching_indices = idx;                                                                           \
                statistics->count_satisfying_values = statistics->count_satisfying_values +                            \
                                                      ((!tuplet_is_null &&                                             \
                                                      (__builtin_expect((values[idx] opp compare_value),               \
                                                      hint_expected_true))));                                          \
                statistics->count_null_values = statistics->count_null_values + tuplet_is_null;                        \
                statistics->count_non_satisfying_values = statistics->count_non_satisfying_values +                    \
                                                          (!tuplet_is_null &&                                          \
                                                          !(__builtin_expect((values[idx] opp compare_value),          \
                                                          hint_expected_true)));                                       \
                skip_for_null = tuplet_is_null && null_policy == null_value_filter_policy::skip_null_values;           \
                *out_matching_indices = ((*out_matching_indices) * (1 - skip_for_null)) + (idx * skip_for_null);       \
                out_matching_indices = out_matching_indices + ( (!tuplet_is_null &&                                    \
                                       (__builtin_expect((values[idx] opp compare_value),                              \
                                       hint_expected_true))) || skip_for_null);                                        \
                statistics->count_skipped_values = statistics->count_skipped_values + skip_for_null;                   \
                ++idx;                                                                                                 \
            }                                                                                                          \
            while (num_elements - idx) {                                                                               \
                SMART_BITMASK_GET_UNSAFE_FAST(tuplet_is_null, null_mask, idx);                                         \
                *out_matching_indices = idx;                                                                           \
                statistics->count_satisfying_values = statistics->count_satisfying_values + ((!tuplet_is_null &&       \
                                                      (__builtin_expect((values[idx] opp compare_value),               \
                                                      hint_expected_true))));                                          \
                statistics->count_null_values = statistics->count_null_values + tuplet_is_null;                        \
                statistics->count_non_satisfying_values = statistics->count_non_satisfying_values +                    \
                                                          (!tuplet_is_null &&                                          \
                                                          !(__builtin_expect((values[idx] opp compare_value),          \
                                                          hint_expected_true)));                                       \
                skip_for_null = tuplet_is_null && null_policy == null_value_filter_policy::skip_null_values;           \
                *out_matching_indices = ((*out_matching_indices) * (1 - skip_for_null)) + (idx * skip_for_null);       \
                out_matching_indices = out_matching_indices + ((!tuplet_is_null &&                                     \
                                       (__builtin_expect((values[idx] opp compare_value),                              \
                                       hint_expected_true))) || skip_for_null);                                        \
                statistics->count_skipped_values = statistics->count_skipped_values + skip_for_null;                   \
                ++idx;                                                                                                 \
            }                                                                                                          \
        }                                                                                                              \
        *out_num_matching_indices = (out_matching_indices - out_matching_indices_start);                               \
    }                                                                                                                  \
};

namespace mondrian
{
    namespace vpipes
    {
        namespace predicates
        {
            template<class ValueType>
            struct batched_predicates
            {
                using value_t = ValueType;
                using func_t = std::function<void(__out__ size_t *matching_indices,
                                                  __out__ size_t *num_matching_indices,
                                                  __out__ statistics::predicate_run *predicate_statistics,
                                                  __in__ null_value_filter_policy null_policy,
                                                  __in__ const tuplet_id_t *tupletids,
                                                  __in__ const value_t *values,
                                                  __in__ const mtl::smart_bitmask *bitmask,
                                                  __in__ size_t num_elements)>;

                struct less_than
                {
                    DEFINE_STRAIGHT_FORWARD(straightforward_impl, <);
                    DEFINE_MICRO_OPTIMIZED_PREDICATE(micro_optimized_impl, <);
                    BRANCH_FREE_PREDICATE(branch_free_predicate, <);
                    MICRO_BRANCH_FREE_PREDICATE(micro_branch_free_predicate,<);
                    MICRO_UNROLLED_BRANCH_FREE_PREDICATE(micro_unrolled_branch_free_predicate,<);
                };

                struct less_equal
                {
                    DEFINE_STRAIGHT_FORWARD(straightforward_impl, <=);
                    DEFINE_MICRO_OPTIMIZED_PREDICATE(micro_optimized_impl, <=);
                    BRANCH_FREE_PREDICATE(branch_free_predicate, <=);
                    MICRO_BRANCH_FREE_PREDICATE(micro_branch_free_predicate,<=);
                    MICRO_UNROLLED_BRANCH_FREE_PREDICATE(micro_unrolled_branch_free_predicate,<=);
                };

                struct equal_to
                {
                    DEFINE_STRAIGHT_FORWARD(straightforward_impl, ==);
                    DEFINE_MICRO_OPTIMIZED_PREDICATE(micro_optimized_impl, ==);
                    BRANCH_FREE_PREDICATE(branch_free_predicate, ==);
                    MICRO_BRANCH_FREE_PREDICATE(micro_branch_free_predicate,==);
                    MICRO_UNROLLED_BRANCH_FREE_PREDICATE(micro_unrolled_branch_free_predicate,==);
                };

                struct unequal_to
                {
                    DEFINE_STRAIGHT_FORWARD(straightforward_impl, !=);
                    DEFINE_MICRO_OPTIMIZED_PREDICATE(micro_optimized_impl, !=);
                    BRANCH_FREE_PREDICATE(branch_free_predicate, !=);
                    MICRO_BRANCH_FREE_PREDICATE(micro_branch_free_predicate,!=);
                    MICRO_UNROLLED_BRANCH_FREE_PREDICATE(micro_unrolled_branch_free_predicate,!=);
                };

                struct greater_equal
                {
                    DEFINE_STRAIGHT_FORWARD(straightforward_impl, >=);
                    DEFINE_MICRO_OPTIMIZED_PREDICATE(micro_optimized_impl, >=);
                    BRANCH_FREE_PREDICATE(branch_free_predicate, >=);
                    MICRO_BRANCH_FREE_PREDICATE(micro_branch_free_predicate,>=);
                    MICRO_UNROLLED_BRANCH_FREE_PREDICATE(micro_unrolled_branch_free_predicate,>=);
                };

                struct greater_than
                {
                    DEFINE_STRAIGHT_FORWARD(straightforward_impl, >);
                    DEFINE_MICRO_OPTIMIZED_PREDICATE(micro_optimized_impl, >);
                    BRANCH_FREE_PREDICATE(branch_free_predicate, >);
                    MICRO_BRANCH_FREE_PREDICATE(micro_branch_free_predicate,>);
                    MICRO_UNROLLED_BRANCH_FREE_PREDICATE(micro_unrolled_branch_free_predicate,>);
                };
            };
        }
    }
}

