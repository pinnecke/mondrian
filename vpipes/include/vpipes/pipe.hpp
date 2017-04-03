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
        template<class Input, class Output, class InputTupletIdType = size_t, class OutputTupletIdType = size_t>
        class pipe : public consumer<Input, InputTupletIdType>, public producer<Output, OutputTupletIdType>
        {
            using input_super = consumer<Input, InputTupletIdType>;
            using output_super = producer<Output, OutputTupletIdType>;

        public:
            using typename input_super::input_t;
            using typename input_super::input_tupletid_t;
            using typename input_super::input_batch_t;

            using typename output_super::output_t;
            using typename output_super::output_tupletid_t;
            using typename output_super::output_batch_t;
            using typename output_super::consumer_t;

        protected:
            using output_super::produce;

        public:
            pipe(consumer_t *destination, unsigned batch_size):
                    output_super(destination, batch_size) { }

            virtual void close() final
            {
                output_super::close();
            }
        };
    }
}
