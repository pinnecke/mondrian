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
        template<class Output, class OutputForwardIt = Output*>
        class producer
        {
        public:
            using output_t = Output;
            using output_iterator_t = OutputForwardIt;
            using consumer_t = consumer<output_t, output_iterator_t>;
            using output_vector_t = vector<output_t, output_iterator_t>;

        private:
            consumer_t *consumer;
            output_vector_t *result = nullptr;
            size_t size;
        protected:

            void reset() { result = new output_vector_t(size); }

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

            virtual void produce(output_iterator_t *value) final
            {
                produce(value, value + 1);
            }

        protected:

            virtual inline void produce(output_iterator_t *begin, output_iterator_t *end) final
            {
                do {
                    typename output_vector_t::state vector_state;
                    begin = result->add(&vector_state, begin, end);
                    if (vector_state == output_vector_t::state::full) {
                        send();
                        reset();
                    }
                } while (begin != end);
            }

        public:
            producer(consumer_t *consumer, unsigned vector_size):
                    consumer(consumer), size(vector_size)
            {
                reset();
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
        };
    }
}
