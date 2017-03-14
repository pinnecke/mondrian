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

#include "../consumer.hpp"

namespace mondrian
{
    namespace vpipes
    {
        namespace toolkit
        {
            template<class Input, class InputTupletIdType = size_t>
            class materialize : public consumer<Input, InputTupletIdType>
            {
                using super = consumer<Input, InputTupletIdType>;

            public:
                using typename super::input_t;
                using typename super::input_tupletid_t;
                using typename super::materializer_t;

            private:
                size_t i;
                input_t *destination;
                size_t *result_set_size;
                unsigned expected_chunk_size;

            public:
                materialize(Input *destination, size_t *result_set_size, materializer_t materialize_func, unsigned expected_chunk_size) :
                        super(materialize_func), destination(destination), i(0), result_set_size(result_set_size),
                        expected_chunk_size(expected_chunk_size)
                {
                    assert (expected_chunk_size > 0);
                };

            protected:
                inline virtual void on_consume(input_tupletid_t *begin,
                                               input_tupletid_t *end) override final __attribute__((always_inline))
                {
                    size_t distance = (end - begin);
                    super::lookup(destination + i, destination + i + distance, begin, end);
                    i += distance;
                    *result_set_size = i;
                }
            };
        }
    }
}

