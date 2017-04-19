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
            template<class Input, class Output>
            class map : public pipe<Input, Output>
            {
                using super = pipe<Input, Output>;
            public:
                using typename super::input_t;
                using typename super::input_batch_t;
                using typename super::output_t;
                using typename super::output_batch_t;

                using typename super::consumer_t;
                using iterator_t = vpipes::iterator<tuplet_id_t *>;
                using map_func_t = typename vpipes::maps::batched_map<input_t, output_t>::func_t;

            private:
                output_t *value_buffer;
                mtl::smart_bitmask null_mask_buffer;
                size_t buffer_size;
                map_func_t map_func;

            public:

                map(consumer_t *destination, map_func_t map_func, unsigned batch_size) :
                        super(destination, batch_size), map_func(map_func), null_mask_buffer(batch_size)
                {
                    // Note here: The operator is unaware of the batch size of the input. The assignment
                    // of the batch size of this operator as the batch size of the preceding operator
                    // just a best guess and must be corrected afterwards if it was wrong
                    buffer_size = batch_size;
                    value_buffer = (output_t *) malloc(buffer_size * sizeof(output_t));
                    assert (value_buffer != nullptr);
                }

                inline virtual void on_consume(const input_batch_t *data) override final __attribute__((always_inline))
                {
                    auto input_batch_size = data->get_size();

                    if (__builtin_expect(input_batch_size > buffer_size, false)) {
                        buffer_size = input_batch_size;
                        value_buffer = (output_t *) realloc(value_buffer, input_batch_size * sizeof(output_t));
                        assert (value_buffer != nullptr);
                    }

                    null_mask_buffer.unset_all();
                    map_func(value_buffer, &null_mask_buffer, data->get_values(), data->get_null_mask(),
                             data->get_null_info(), input_batch_size);

                    super::produce(data->get_tupletids(), value_buffer, &null_mask_buffer, input_batch_size, true);
                }

                virtual void on_cleanup() override
                {
                    free (value_buffer);
                }
            };
        }
    }
}

