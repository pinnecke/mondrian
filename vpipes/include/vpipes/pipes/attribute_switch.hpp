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
        namespace pipes
        {
            template<class Input, class Output>
            class attribute_switch : public pipe<Input, Output>
            {
                using super = pipe<Input, Output>;
            public:
                using typename super::input_t;
                using typename super::input_batch_t;
                using typename super::output_t;
                using typename super::output_batch_t;
                using typename super::consumer_t;

                using point_copy_func_t = typename point_copy<output_t>::func_t;
                using point_null_copy_func_t = typename point_null_copy::func_t;

            private:
                point_copy_func_t point_copy;
                point_null_copy_func_t m_point_null_copy;

                output_t *out_projected_values;
                mtl::smart_bitmask out_projected_bitmask;
                size_t buffer_size;
            public:

                attribute_switch(__in__ consumer_t *destination,
                        __in__ point_copy_func_t point_copy,
                        __in__ point_null_copy_func_t point_null_copy,
                        __in__ unsigned batch_size) :
                        super(destination, batch_size), out_projected_bitmask(batch_size),
                        point_copy(point_copy), m_point_null_copy(point_null_copy)
                {
                    // Note here: The operator is unaware of the batch size of the input. The assignment
                    // of the batch size of this operator as the batch size of the preceding operator
                    // just a best guess and must be corrected afterwards if it was wrong
                    buffer_size = batch_size;
                    out_projected_values = (output_t *) malloc(buffer_size * sizeof(output_t));
                    assert (out_projected_values != nullptr);
                }

                inline virtual void on_consume(__in__ const input_batch_t *data) override final __attribute__((always_inline))
                {
                    auto input_batch_size = data->get_size();
                    auto in_tupletids = data->get_tupletids();
                    auto in_values = data->get_values();

                    if (__builtin_expect(input_batch_size > buffer_size, false)) {
                        buffer_size = input_batch_size;
                        out_projected_values = (output_t *) realloc(out_projected_values, input_batch_size *
                                                                                              sizeof(output_t));
                        assert (out_projected_values != nullptr);
                    }

                    out_projected_bitmask.unset_all();
                    m_point_null_copy(&out_projected_bitmask, in_tupletids, input_batch_size);
                    point_copy(out_projected_values, in_tupletids, input_batch_size);
                    super::produce(in_tupletids, out_projected_values, &out_projected_bitmask, input_batch_size, false);
                }

                virtual void on_cleanup() override
                {
                    free (out_projected_values);
                }

                virtual const char *get_class_name() const override
                {
                    return "vpipes::pipes::attribute_switch";
                }
            };
        }
    }
}

