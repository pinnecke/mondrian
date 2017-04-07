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
            consumer_t **destinations;
            output_batch_t *result = nullptr;
            size_t batch_size, num_destintations;

        protected:
            void add_destination(consumer_t *destination)
            {
                assert (destination != nullptr);
                destinations = (consumer_t **) realloc(destinations, ++num_destintations * sizeof(consumer_t*));
                assert (destinations != nullptr);
                destinations[num_destintations - 1] = destination;
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
                assert (destinations != nullptr);
                result->prefetch(cpu_hint::for_read);
                for (size_t idx = 0; idx < num_destintations; ++idx) {
                    destinations[idx]->consume(result);
                }
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
                    size_t this_batch_size = MIN(batch_size, end - offset);
                    result->iota(offset, this_batch_size, block_copy_func);
                    send();
                    offset += this_batch_size;
                }
            }

        protected:

            virtual inline void produce(const output_tupletid_t *tupletids, const output_t *values,
                                        const mtl::smart_bitmask *null_mask,
                                        const size_t *indices, size_t num_indices,
                                        bool hint_hit_out_batch_size) final __attribute__((always_inline))
            {
                auto total = num_indices, remaining = num_indices;
                do {
                    typename output_batch_t::state batch_state;
                    auto step = (total - remaining);
                    result->prefetch(cpu_hint::for_write);
                    remaining = result->add(&batch_state, tupletids, values, null_mask, indices + step, remaining);
                    if (__builtin_expect(batch_state == output_batch_t::state::full, hint_hit_out_batch_size)) {
                        send();
                    }
                } while (remaining);
            }

            virtual inline void produce(const output_tupletid_t *tupletids, const output_t * values,
                                        const mtl::smart_bitmask *null_mask,
                                        size_t num_elements, bool hint_hit_out_batch_size)
                                        final __attribute__((always_inline))
            {
                auto total_num = num_elements, remaining_num = num_elements;
                do {
                    result->prefetch(cpu_hint::for_write);

                    typename output_batch_t::state batch_state;
                    auto step = (total_num - remaining_num);
                    const_cast<mtl::smart_bitmask *>(null_mask)->set_offset(step);
                    remaining_num = result->add(&batch_state, tupletids + step, values + step,
                                                null_mask, remaining_num);
                    if (__builtin_expect(batch_state == output_batch_t::state::full, hint_hit_out_batch_size)) {
                        send();
                    }
                } while (remaining_num);
                const_cast<mtl::smart_bitmask *>(null_mask)->set_offset(0);
            }

            virtual void close_consumers()
            {
                for (size_t idx = 0; idx != num_destintations; ++idx) {
                    consumer_t *dest = destinations[idx];
                    if (__builtin_expect(dest != nullptr, true)) {
                        dest->close();
                    }
                }
            }

            virtual void close()
            {
                send();
                on_close();
                send();
                close_consumers();
                cleanup();
                on_cleanup();
            }

        public:
            producer(consumer_t *destination, unsigned batch_size):
                    num_destintations(0), batch_size(batch_size)
            {
                result = new output_batch_t(batch_size);

                destinations = (consumer_t **) malloc (sizeof(consumer_t*));
                if (destination != nullptr) {
                    add_destination(destination);
                }

                assert (destinations != nullptr);
            }

            inline virtual void start() final
            {
                on_start();
                close();
            }
        };
    }
}
