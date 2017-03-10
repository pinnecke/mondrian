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
#include "../pipe.hpp"

using namespace std;

namespace mondrian
{
    namespace vpipes
    {
        namespace toolkit
        {
            template<class Input, class InputTupletIdType = size_t>
            class batched_pred_filter : public pipe<Input, Input, InputTupletIdType, InputTupletIdType>
            {
                using super = pipe<Input, Input, InputTupletIdType, InputTupletIdType>;
            public:
                using typename super::input_t;
                using typename super::input_tupletid_t;
                using typename super::consumer_t;
                using typename super::materializer_t;
                using iterator_t = vpipes::iterator<input_tupletid_t *>;
                using predicate_t = typename vpipes::functional::batched_predicates<input_t>::func_t;

            private:
                input_tupletid_t *result_buffer;
                input_t *value_buffer;
                size_t buffer_size;

                predicate_t predicate;
            public:

                batched_pred_filter(consumer_t *consumer, materializer_t materializer, predicate_t predicate, unsigned chunk_size) :
                        super(consumer, materializer, chunk_size), predicate(predicate)
                {
                    // Note here: The operator is unaware of the vector size of the input. The assignment
                    // of the vector size of this operator as the vector size of the preceding operator
                    // just a best guess and must be corrected afterwards if it was wrong
                    buffer_size = chunk_size;
                    result_buffer = (input_tupletid_t *) malloc(buffer_size * sizeof(input_tupletid_t));
                    value_buffer = (input_t *) malloc(buffer_size * sizeof(input_t));
                    assert (result_buffer != nullptr);
                    assert (value_buffer != nullptr);
                }

                virtual void on_consume(input_tupletid_t *begin, input_tupletid_t *end) override
                {
                    auto input_chunk_size = (end - begin);

                    if (input_chunk_size > buffer_size) {
                        buffer_size = input_chunk_size;
                        result_buffer = (input_tupletid_t *) realloc(result_buffer, input_chunk_size *
                                                                     sizeof(input_tupletid_t));
                        value_buffer = (input_t *) realloc(value_buffer, input_chunk_size *
                                                                     sizeof(input_t));
                        assert (result_buffer != nullptr);
                        assert (value_buffer != nullptr);
                    }
                    size_t result_size = 0;
                    super::lookup(value_buffer, value_buffer + input_chunk_size, begin, end);
                    predicate(result_buffer, &result_size, begin, end, value_buffer, value_buffer + input_chunk_size);
                    super::produce(result_buffer, result_buffer + result_size);
                }

                virtual void on_cleanup() override
                {
                    free (result_buffer);
                    free (value_buffer);
                }
            };

            template<class Type, class InputTupletIdType = size_t>
            class simple_filter : public pipe<Type, Type, InputTupletIdType, InputTupletIdType>
            {
                using super = pipe<Type, Type, InputTupletIdType, InputTupletIdType>;
            public:
                using typename super::input_t;
                using typename super::input_tupletid_t;
                using typename super::consumer_t;
                using predicate_t = std::function<bool(input_t *value)>;

            private:
                predicate_t predicate;
            public:

                simple_filter(consumer_t *consumer, unsigned chunk_size, predicate_t predicate) :
                        super(consumer, chunk_size), predicate(predicate) { }

                virtual void on_consume(input_tupletid_t *begin, input_tupletid_t *end) override
                {
                    for (auto it = begin; it != end; ++it) {
                        if (predicate(&lookup(it)))
                            super::produce(it);
                    }
                }
            };
        }
    }
}

