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

#define DEFINE_INDICATE_STRAIGHT_FORWARD(name, opp)                                                                    \
struct name                                                                                                            \
{                                                                                                                      \
    input_t compare_value;                                                                                             \
                                                                                                                       \
    name(input_t compare_value): compare_value(compare_value) { }                                                      \
                                                                                                                       \
    void operator()(output_t *destination, const input_t *source, size_t num_elements)                                 \
    {                                                                                                                  \
        assert (destination != nullptr);                                                                               \
        assert (source != nullptr);                                                                                    \
        while (num_elements--) {                                                                                       \
            *destination++ = (*source++ opp compare_value);                                                            \
        }                                                                                                              \
    }                                                                                                                  \
};

namespace mondrian
{
    namespace vpipes
    {
        namespace maps
        {
            template<class Input, class Output>
            struct batched_map
            {
                using input_t = Input;
                using output_t = Output;

                using func_t = std::function<void(output_t *destination, const input_t *source, size_t num_elements)>;
            };

            template<class Input>
            struct indicators : public batched_map<Input, bool>
            {
            private:
                using super = batched_map<Input, bool>;

            public:
                using typename super::input_t;
                using typename super::output_t;
                using typename super::func_t;

                struct less_than
                {
                    DEFINE_INDICATE_STRAIGHT_FORWARD(straightforward_impl, <);
                };

                struct less_equal
                {
                    DEFINE_INDICATE_STRAIGHT_FORWARD(straightforward_impl, <=);
                };

                struct equal_to
                {
                    DEFINE_INDICATE_STRAIGHT_FORWARD(straightforward_impl, ==);
                };

                struct unequal_to
                {
                    DEFINE_INDICATE_STRAIGHT_FORWARD(straightforward_impl, !=);
                };

                struct greater_equal
                {
                    DEFINE_INDICATE_STRAIGHT_FORWARD(straightforward_impl, >=);
                };

                struct greater_than
                {
                    DEFINE_INDICATE_STRAIGHT_FORWARD(straightforward_impl, >);
                };
            };
        }
    }
}

