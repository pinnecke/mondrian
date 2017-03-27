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
            struct mark_map : public batched_map<Input, bool>
            {
            private:
                using super = batched_map<Input, bool>;

            public:
                using typename super::input_t;
                using typename super::output_t;
                using typename super::func_t;

                struct mark_less_than
                {
                    struct straightforward_impl
                    {
                        input_t compare_value;

                        straightforward_impl(input_t input_t): compare_value() { }

                        void operator()(output_t *destination, const input_t *source, size_t num_elements)
                        {
                            assert (destination != nullptr);
                            assert (source != nullptr);
                            while (num_elements--) {
                                *destination++ = (*source++ == compare_value);
                            }
                        }
                    };
                };

                struct mark_less_equal
                {
                    // TODO ...
                };

                struct mark_equal_to
                {
                    // TODO ...
                };

                struct mark_unequal_to
                {
                    // TODO ...
                };

                struct mark_greater_equal
                {
                    // TODO ...
                };

                struct mark_greater_than
                {
                    // TODO ...
                };
            };
        }
    }
}

