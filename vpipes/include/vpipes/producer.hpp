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
        template<class Output, class OutputTupletIdType = size_t>
        class producer
        {
        public:
            using output_t = Output;
            using output_tupletid_t = OutputTupletIdType;
            using consumer_t = consumer<output_t, output_tupletid_t>;
            using output_chunk_t = chunk<output_t, output_tupletid_t>;
            using linker_t = typename functional::linker<output_t, output_tupletid_t>::func_t;

        private:
            consumer_t *next_operator;
            output_chunk_t *result = nullptr;
            size_t size;

        protected:
            void set_consumer(consumer_t *next_operator)
            {
                this->next_operator = next_operator;
            }

            void reset() {
                result->reset();
            }

            void cleanup()
            {
                if (result != nullptr) {
                    result->release();
                    delete (result);
                    result = nullptr;
                }
            }

            inline void send() __attribute__((always_inline))
            {
                assert (next_operator != nullptr);
                result->memory_prefetch_for_read();
                next_operator->consume(result);
                reset();
            }

        protected:

            virtual void on_close() { };

            virtual void on_cleanup() { };

            virtual void on_start() { };

            /*inline virtual void produce(output_tupletid_t *value) final __attribute__((always_inline))
            {
                produce(value, value + 1, false);
            }*/

            inline virtual void produce_tupletid_range(output_tupletid_t start, output_tupletid_t end,
                                                       linker_t linker_func) final __attribute__((always_inline))
            {
                assert (start <= end);

                output_tupletid_t offset = start;
                while (offset < end) {
                    size_t this_chunk_size = MIN(size, end - offset);
                    result->iota(offset, this_chunk_size, linker_func);
                    send();
                    offset += this_chunk_size;
                }
            }

        protected:

            virtual inline void produce(const output_tupletid_t *tupletids, output_t * const *values,
                                        const size_t *indices, size_t num_indices,
                                        bool expect_output_chunk_is_full_afterwards) final __attribute__((always_inline))
            {
                result->memory_prefetch_for_write();
                do {
                    typename output_chunk_t::state chunk_state;
                    num_indices = result->add(&chunk_state, tupletids, values, indices, num_indices);
                    if (__builtin_expect(chunk_state == output_chunk_t::state::full,
                                         expect_output_chunk_is_full_afterwards)) {
                        send();
                    }
                } while (num_indices);
            }

//            virtual inline void produce(output_tupletid_t *begin, output_tupletid_t *end,
//                                        bool expect_output_chunk_is_full_afterwards) final __attribute__((always_inline))
//            {
//                result->memory_prefetch_for_write();
//                do {
//                    typename output_chunk_t::state chunk_state;
//                    begin = result->add(&chunk_state, begin, end, );
//                    if (__builtin_expect(chunk_state == output_chunk_t::state::full,
//                                         expect_output_chunk_is_full_afterwards)) {
//                        send();
//                    }
//                } while (begin != end);
//            }

            virtual void close()
            {
                if (__builtin_expect(next_operator != nullptr, true))
                {
                    send();
                    on_close();
                    send();
                    next_operator->close();
                }
                cleanup();
                on_cleanup();
            }

        public:
            producer(consumer_t *next_operator, unsigned chunk_size):
                    next_operator(next_operator), size(chunk_size)
            {
                result = new output_chunk_t(chunk_size);
            }



            inline virtual void start() final {
                on_start();
                close();
            }
        };
    }
}
