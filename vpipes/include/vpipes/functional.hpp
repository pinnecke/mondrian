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

#include <functional>
#include <cassert>

#define ASSERT_VALID_BATCHED_PREDICATE_ARGS()                           \
{                                                                       \
    assert (*result_size == 0);                                         \
    assert (result_buffer != nullptr);                                  \
    assert (values_begin != nullptr && values_end != nullptr);          \
    assert (values_begin <= values_end);                                \
    assert (tupletids_begin != nullptr && tupletids_end != nullptr);    \
    assert (tupletids_begin <= tupletids_end);                          \
}

#define POINTER_DISTANCE(begin, end)                                    \
    (end - begin)

#define DEFINE_STD_BINARY_PREDICATE(name, opp)                                                                  \
struct name                                                                                                     \
{                                                                                                               \
    value_t compare_value;                                                                                      \
                                                                                                                \
    name(value_t compare_value): compare_value(compare_value) { }                                               \
                                                                                                                \
    void operator()(tupletid_t *result_buffer, size_t *result_size,                                             \
                    const tupletid_t *tupletids_begin, const tupletid_t *tupletids_end,                         \
                    const value_t *values_begin, const value_t *values_end)                                     \
    {                                                                                                           \
        ASSERT_VALID_BATCHED_PREDICATE_ARGS();                                                                  \
        for (auto value_it = values_begin; value_it != values_end; ++value_it)                                  \
            if (*value_it opp compare_value)                                                                    \
                result_buffer[(*result_size)++] = tupletids_begin[POINTER_DISTANCE(values_begin, value_it)];    \
    }                                                                                                           \
};

namespace mondrian
{
    namespace vpipes
    {
        namespace functional {

            template <class ValueType, class TupletIdType = size_t>
            struct batched_materializes
            {
                using value_t = ValueType;
                using tupletid_t = TupletIdType;
                using func_t = std::function<void(value_t *out_begin, value_t *out_end,
                                                  const tupletid_t *begin, const tupletid_t *end)>;
            };

            template<class ValueType, class TupletIdType = size_t>
            struct batched_predicates
            {
                using value_t = ValueType;
                using tupletid_t = TupletIdType;
                using func_t = std::function<void(tupletid_t *result_buffer, size_t *result_size,
                                                  const tupletid_t *tupletids_begin, const tupletid_t *tupletids_end,
                                                  const value_t *values_begin, const value_t *values_end)>;

                struct less_than
                {
                    DEFINE_STD_BINARY_PREDICATE(straightforward_impl, <);
                };

                struct less_equal
                {
                    DEFINE_STD_BINARY_PREDICATE(straightforward_impl, <=);
                };

                struct equal_to
                {
                    DEFINE_STD_BINARY_PREDICATE(straightforward_impl, ==);
                };

                struct unequal_to
                {
                    DEFINE_STD_BINARY_PREDICATE(straightforward_impl, !=);
                };

                struct greater_equal
                {
                    DEFINE_STD_BINARY_PREDICATE(straightforward_impl, >=);
                };

                struct greater
                {
                    DEFINE_STD_BINARY_PREDICATE(straightforward_impl, >);
                };
            };

        }


    }
}

