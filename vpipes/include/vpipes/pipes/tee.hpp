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

using namespace std;

namespace mondrian
{
    namespace vpipes
    {
        namespace pipes
        {
            template<class Input, class InputTupletIdType = size_t>
            class tee : public pipe<Input, Input, InputTupletIdType, InputTupletIdType>
            {
                using super = pipe<Input, Input, InputTupletIdType, InputTupletIdType>;
            public:
                using typename super::input_t;
                using typename super::input_tupletid_t;
                using typename super::input_batch_t;
                using typename super::output_t;
                using typename super::output_tupletid_t;
                using typename super::output_batch_t;
                using typename super::consumer_t;

            public:

                tee(consumer_t *destination1, consumer_t *destination2, unsigned batch_size) :
                        super(destination1, batch_size)
                {
                    super::add_destination(destination2);
                }

                inline virtual void on_consume(const input_batch_t *data) override final __attribute__((always_inline))
                {
                    super::produce(data->get_tupletids(), data->get_values(), data->get_size(), true);
                }
            };
        }
    }
}

