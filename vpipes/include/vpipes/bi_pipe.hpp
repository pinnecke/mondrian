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
#include "bi_consumer.hpp"
#include "pipe_tail.hpp"

namespace mondrian
{
    namespace vpipes
    {
        template<class InputLeft, class InputRight, class Output,
                     class InputLeftForwardIt = InputLeft*, class InputRightForwardIt = InputRight*,
                     class OutputForwardIt = Output*>
        class bi_pipe : public bi_consumer<InputLeft, InputRight, InputLeftForwardIt, InputRightForwardIt>,
                               producer<Output, OutputForwardIt>
        {
            using input_super = bi_consumer<InputLeft, InputRight, InputLeftForwardIt, InputRightForwardIt>;
            using output_super = producer<Output, OutputForwardIt>;
        public:
            using typename input_super::input_left_t;
            using typename input_super::input_left_iterator_t;
            using typename input_super::input_left_chunk_t;
            using typename input_super::input_right_t;
            using typename input_super::input_right_iterator_t;
            using typename input_super::input_right_chunk_t;

            using typename output_super::output_t;
            using typename output_super::output_iterator_t;
            using typename output_super::consumer_t;
            using typename output_super::output_chunk_t;

            using output_super::forward;
            using input_super::lookup_left;
            using input_super::lookup_right;
            using input_super::as_reference_left;
            using input_super::as_reference_right;

        public:
            bi_pipe(consumer_t *consumer, unsigned chunk_size):
                    output_super(consumer, chunk_size) { }
        };
    }
}
