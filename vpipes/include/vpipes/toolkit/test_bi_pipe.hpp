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
#include "../bi_pipe.hpp"

using namespace std;

namespace mondrian
{
    namespace vpipes
    {
        namespace toolkit
        {
            template<class InputLeft, class InputRight, class Output,
                        class InputLeftForwardIt = InputLeft*, class InputRightForwardIt = InputRight*,
                        class OutputForwardIt = Output*>
            class test_bi_pipe : public bi_pipe<InputLeft, InputRight, Output, InputLeftForwardIt,
                                                InputRightForwardIt, OutputForwardIt>
            {
                using super = bi_pipe<InputLeft, InputRight, Output, InputLeftForwardIt,
                        InputRightForwardIt, OutputForwardIt>;
            public:
                using typename super::input_left_t;
                using typename super::input_left_iterator_t;
                using typename super::input_left_chunk_t;
                using typename super::input_right_t;
                using typename super::input_right_iterator_t;
                using typename super::input_right_chunk_t;
                using typename super::output_t;
                using typename super::output_iterator_t;
                using typename super::consumer_t;
                using typename super::output_chunk_t;

            public:

                test_bi_pipe(consumer_t *consumer, unsigned chunk_size) :
                            super(consumer, chunk_size) { }

                virtual void on_consume_left(input_left_iterator_t *begin, input_left_iterator_t *end) override {
                    // TODO: nested loop + full-consume (using .close() to call close(Left/Right) + delayed vector deallocation

                    std::cout << "consume left: " << (end - begin) << std::endl;
                }

                virtual void on_consume_right(input_right_iterator_t *begin, input_right_iterator_t *end) override {
                    std::cout << "consume right: " << (end - begin) << std::endl;
                }
            };
        }
    }
}

