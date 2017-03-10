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

#include "pipe_tail.hpp"

namespace mondrian
{
    namespace vpipes
    {
        template<class Input, class Output, class InputTupletIdType = size_t, class OutputTupletIdType = size_t>
        class batch_pipe : public consumer<Input, InputTupletIdType>
        {
            using super = consumer<Input, InputTupletIdType>;
        public:
            using typename super::input_t;
            using typename super::input_tupletid_t;
            using typename super::input_chunk_t;

            using output_t = Output;
            using output_tupletid_t = OutputTupletIdType;
            using consumer_t = consumer<output_t, output_tupletid_t>;

        private:
            consumer_t *consumer;
            input_tupletid_t *batch = nullptr;
            size_t size, capacity;

            void cleanup()
            {
                free (batch);
                batch = nullptr;
            }

        protected:
            virtual void on_consume(input_tupletid_t *begin, input_tupletid_t *end) final override
            {
                auto dist = distance(begin, end);
                auto next_size = dist + size + 1;
                if (next_size >= capacity) {
                    while (next_size >= capacity)
                        capacity *= 1.4f;
                    batch = (input_tupletid_t *) realloc(batch, capacity * sizeof(input_tupletid_t));
                    assert (batch != nullptr);
                }
                for (auto it = begin; it < end; ++it) {
                    batch[size++] = *it;
                }
            }

            virtual void on_batch_process(output_tupletid_t **output_begin, output_tupletid_t **output_end,
                                          input_tupletid_t *begin, input_tupletid_t *end) = 0;

            virtual void on_cleanup() { };

            virtual void close() override final
            {
                if (consumer != nullptr)
                {
                    auto input_begin = batch;
                    auto input_end = input_begin + size;
                    output_tupletid_t *output_begin, *output_end;

                    on_batch_process(&output_begin, &output_end, input_begin, input_end);
                    assert (output_begin != nullptr);
                    assert (output_end != nullptr);
                    assert (output_begin <= output_end);

                    consumer->consume(output_begin, output_end);
                    consumer->close();
                }
                on_cleanup();
                cleanup();
            }

        public:
            batch_pipe(consumer_t *consumer, unsigned initial_capacity):
                    consumer(consumer), capacity(initial_capacity), size(0)
            {
                batch = (input_tupletid_t *) malloc (initial_capacity * sizeof(input_tupletid_t));
                assert (batch != nullptr);
            }
        };
    }
}
