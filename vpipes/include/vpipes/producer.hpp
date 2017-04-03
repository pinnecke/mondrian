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
            using output_batch_t = batch<output_t, output_tupletid_t>;
            using block_copy_t = typename block_copy<output_t, output_tupletid_t>::func_t;

        private:
            consumer_t *next_operator;
            output_batch_t *result = nullptr;
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

            inline virtual void produce_tupletid_range(output_tupletid_t start, output_tupletid_t end,
                                                       block_copy_t block_copy_func) final __attribute__((always_inline))
            {
                assert (start <= end);

                output_tupletid_t offset = start;
                while (offset < end) {
                    size_t this_batch_size = MIN(size, end - offset);
                    result->iota(offset, this_batch_size, block_copy_func);
                    send();
                    offset += this_batch_size;
                }
            }

        protected:

            virtual inline void produce(const output_tupletid_t *tupletids, const output_t * values,
                                        const size_t *indices, size_t num_indices,
                                        bool expect_output_batch_is_full_afterwards) final __attribute__((always_inline))
            {
                auto original_num_indices =num_indices;
                result->memory_prefetch_for_write();
                do {
                    typename output_batch_t::state batch_state;
                    num_indices = result->add(&batch_state, tupletids, values, indices, num_indices);
                    if (__builtin_expect(batch_state == output_batch_t::state::full,
                                         expect_output_batch_is_full_afterwards)) {
                        send();
                    }
                    indices = indices + (original_num_indices  - num_indices);
                } while (num_indices);
            }

            virtual inline void produce(const output_tupletid_t *tupletids, const output_t * values,
                                        size_t num_elements, bool expect_output_batch_is_full_afterwards)
                                        final __attribute__((always_inline))
            {
                auto original_num_elements = num_elements;
                result->memory_prefetch_for_write();
                do {
                    typename output_batch_t::state batch_state;
                    num_elements = result->add(&batch_state, tupletids, values, num_elements);
                    if (__builtin_expect(batch_state == output_batch_t::state::full,
                                         expect_output_batch_is_full_afterwards)) {
                        send();
                    }
                    auto step = (original_num_elements - num_elements);
                    tupletids += step;
                    tupletids += step;
                } while (num_elements);
            }

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
            producer(consumer_t *next_operator, unsigned batch_size):
                    next_operator(next_operator), size(batch_size)
            {
                result = new output_batch_t(batch_size);
            }

            inline virtual void start() final
            {
                on_start();
                close();
            }
        };
    }
}
