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


#define DEFINE_JOIN_CONDITION(name, opp)                                                                               \
struct name                                                                                                            \
{                                                                                                                      \
                                                                                                                       \
    explicit name() { }                                                                                               \
                                                                                                                       \
    virtual inline bool operator()(__in__ value_t left_value,                                                          \
                                   __in__ value_t right_value) final __attribute__((always_inline))                    \
    {                                                                                                                  \
                return left_value opp right_value;                                                                     \
    }                                                                                                                  \
};

#define DEFINE_CHARTISIAN_PRODUCT(name)                                                                                \
struct name                                                                                                            \
{                                                                                                                      \
                                                                                                                       \
    explicit name() { }                                                                                                \
                                                                                                                       \
    virtual inline bool operator()(__in__ value_t left_value,                                                          \
                                   __in__ value_t right_value) final __attribute__((always_inline))                    \
    {                                                                                                                  \
                return true;                                                                                           \
    }                                                                                                                  \
};


namespace mondrian
{
    namespace vpipes
    {
        namespace predicates
        {
            template<class ValueType>
            struct join_conditions
            {
                using value_t = ValueType;

                using func_t = std::function<bool(__in__ value_t left_value,
                                                  __in__ value_t right_value)>;

                struct less_than
                {
                    DEFINE_JOIN_CONDITION(join_condition_impl, <);
                };

                struct less_equal
                {
                    DEFINE_JOIN_CONDITION(join_condition_impl, <=);
                };

                struct equal_to
                {
                    DEFINE_JOIN_CONDITION(join_condition_impl, ==);
                };

                struct unequal_to
                {
                    DEFINE_JOIN_CONDITION(join_condition_impl, !=);
                };

                struct greater_equal
                {
                    DEFINE_JOIN_CONDITION(join_condition_impl, >=);
                };

                struct greater_than
                {
                    DEFINE_JOIN_CONDITION(join_condition_impl, >);
                };

                struct chartisian_product
                {
                    DEFINE_CHARTISIAN_PRODUCT(cartisian_prod);
                };

            };
        }
    }
}

