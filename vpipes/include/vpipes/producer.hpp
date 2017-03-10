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
        template<class Output, class OutputTupletIdType = size_t>
        class producer
        {
        public:
            using output_t = Output;
            using output_tupletid_t = OutputTupletIdType;
            using consumer_t = consumer<output_t, output_tupletid_t>;
            using output_chunk_t = chunk<output_t, output_tupletid_t>;

        private:
            consumer_t *consumer;
            output_chunk_t *result = nullptr;
            size_t size;
        protected:

            void reset() { result = new output_chunk_t(size); }

            void cleanup()
            {
                if (result != nullptr) {
                    result->release();
                    delete (result);
                    result = nullptr;
                }
            }

            void send()
            {
                consumer->consume(result);
                cleanup();
            }

        protected:

            virtual void on_close() { };

            virtual void on_cleanup() { };

            virtual void on_start() { };

            virtual void produce(output_tupletid_t *value) final
            {
                produce(value, value + 1);
            }

        protected:

            virtual inline void produce(output_tupletid_t *begin, output_tupletid_t *end) final
            {
                do {
                    typename output_chunk_t::state chunk_state;
                    begin = result->add(&chunk_state, begin, end);
                    if (chunk_state == output_chunk_t::state::full) {
                        send();
                        reset();
                    }
                } while (begin != end);
            }

            virtual void close()
            {
                if (consumer != nullptr)
                {
                    send();
                    reset();
                    on_close();
                    send();
                    consumer->close();
                }
                cleanup();
                on_cleanup();
            }

        public:
            producer(consumer_t *consumer, unsigned chunk_size):
                    consumer(consumer), size(chunk_size)
            {
                reset();
            }



            virtual void start() final {
                on_start();
                close();
            }
        };
    }
}
