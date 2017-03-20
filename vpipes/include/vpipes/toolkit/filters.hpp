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
        namespace toolkit
        {
            template<class Input, class InputTupletIdType = size_t>
            class filter : public pipe<Input, Input, InputTupletIdType, InputTupletIdType>
            {
                using super = pipe<Input, Input, InputTupletIdType, InputTupletIdType>;
            public:
                using typename super::input_t;
                using typename super::input_tupletid_t;
                using typename super::output_t;
                using typename super::output_tupletid_t;
                using typename super::consumer_t;
                using typename super::input_chunk_t;
                using iterator_t = vpipes::iterator<input_tupletid_t *>;
                using predicate_t = typename vpipes::functional::batched_predicates<input_t>::func_t;

            private:
                size_t *matching_indices_buffer;
                size_t buffer_size;

                predicate_t predicate;
            public:

                filter(consumer_t *consumer, predicate_t predicate, unsigned chunk_size) :
                        super(consumer, chunk_size), predicate(predicate)
                {
                    // Note here: The operator is unaware of the vector size of the input. The assignment
                    // of the vector size of this operator as the vector size of the preceding operator
                    // just a best guess and must be corrected afterwards if it was wrong
                    buffer_size = chunk_size;
                    matching_indices_buffer = (size_t *) malloc(buffer_size * sizeof(size_t));
                    assert (matching_indices_buffer != nullptr);
                }

                inline virtual void on_consume(const input_chunk_t *data) override final __attribute__((always_inline))
                {
                    auto input_chunk_size = data->get_size();

                    if (__builtin_expect(input_chunk_size > buffer_size, false)) {
                        buffer_size = input_chunk_size;
                        matching_indices_buffer = (size_t *) realloc(matching_indices_buffer, input_chunk_size *
                                                                     sizeof(input_tupletid_t));
                        assert (matching_indices_buffer != nullptr);
                    }

                    size_t result_size = 0;
                    predicate(matching_indices_buffer, &result_size,
                              data->get_tupletids_begin(), data->get_values_begin(), data->get_size());
                    assert (result_size <= buffer_size);
                    super::produce(data->get_tupletids_begin(), data->get_values_begin(), matching_indices_buffer,
                                   result_size, false);
                }

                virtual void on_cleanup() override
                {
                    free (matching_indices_buffer);
                }
            };
        }
    }
}

