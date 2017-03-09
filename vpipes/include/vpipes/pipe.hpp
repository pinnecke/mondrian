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

#include "producer.hpp"
#include "pipe_tail.hpp"

namespace mondrian
{
    namespace vpipes
    {
        template<class Input, class Output, class InputForwardIt = Input*, class OutputForwardIt = Output*>
        class pipe : public consumer<Input, InputForwardIt>, public producer<Output, OutputForwardIt>
        {
            using input_super = consumer<Input, InputForwardIt>;
            using output_super = producer<Output, OutputForwardIt>;

        public:
            using typename input_super::input_t;
            using typename input_super::input_iterator_t;
            using typename input_super::input_chunk_t;

            using typename output_super::output_t;
            using typename output_super::output_iterator_t;
            using typename output_super::output_chunk_t;
            using typename output_super::consumer_t;

        protected:
            using output_super::produce;
            using input_super::lookup;
            using input_super::as_reference;

        public:
            pipe(consumer_t *consumer, unsigned chunk_size):
                    output_super(consumer, chunk_size) { }

            virtual void close() final
            {
                output_super::close();
            }
        };
    }
}
