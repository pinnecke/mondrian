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

#include "../batch_pipe.hpp"

using namespace std;

namespace mondrian
{
    namespace vpipes
    {
        namespace toolkit
        {
            template<class Input, class InputForwardIt = Input*>
            class collector : public batch_pipe<Input, Input, InputForwardIt, InputForwardIt>
            {
                using super = batch_pipe<Input, Input, InputForwardIt, InputForwardIt>;
            public:
                using typename super::input_t;
                using typename super::input_iterator_t;
                using typename super::consumer_t;

            protected:
                virtual void on_batch_process(input_iterator_t **output_begin, input_iterator_t **output_end,
                                              input_iterator_t *begin, input_iterator_t *end) override
                {
                    assert (begin != nullptr);
                    assert (end != nullptr);
                    assert (begin <= end);

                    *output_begin = begin;
                    *output_end = end;
                }

            public:
                collector(consumer_t *consumer, unsigned initial_capacity = 10000) :
                    super(consumer, initial_capacity) { }
            };
        }
    }
}