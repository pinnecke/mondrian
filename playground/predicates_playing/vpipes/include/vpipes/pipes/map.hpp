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

using namespace std;

namespace mondrian
{
    namespace vpipes
    {
        namespace pipes
        {
            template<class Input, class Output, class InputTupletIdType = size_t, class OutputTupletIdType = size_t>
            class map : public pipe<Input, Output, InputTupletIdType, OutputTupletIdType>
            {
                using super = pipe<Input, Output, InputTupletIdType, OutputTupletIdType>;
            public:
                using typename super::input_t;
                using typename super::input_tupletid_t;
                using typename super::input_batch_t;
                using typename super::output_t;
                using typename super::output_tupletid_t;
                using typename super::output_batch_t;

                using typename super::consumer_t;
                using iterator_t = vpipes::iterator<input_tupletid_t *>;
                using map_func_t = typename vpipes::maps::batched_map<input_t, output_t>::func_t;

            private:
                map_func_t map_func;

            public:

                map(consumer_t *consumer, map_func_t map_func, unsigned batch_size) :
                        super(consumer, batch_size), map_func(map_func) { }

                inline virtual void on_consume(const input_batch_t *data) override final __attribute__((always_inline))
                {


                    /*auto input_batch_size = data->get_size();

                    if (__builtin_expect(input_batch_size > buffer_size, false)) {
                        buffer_size = input_batch_size;
                        matching_indices_buffer = (size_t *) realloc(matching_indices_buffer, input_batch_size *
                                                                                              sizeof(input_tupletid_t));
                        assert (matching_indices_buffer != nullptr);
                    }

                    size_t result_size = 0;
                    predicate(matching_indices_buffer, &result_size,
                              data->get_tupletids_begin(), data->get_values_begin(), data->get_size());
                    assert (result_size <= buffer_size);
                    super::produce(data->get_tupletids_begin(), data->get_values_begin(), matching_indices_buffer,
                                   result_size, false);*/
                }
            };
        }
    }
}

