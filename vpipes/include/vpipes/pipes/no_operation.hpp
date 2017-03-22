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
        namespace pipes
        {
            template<class Input, class InputTupletIdType = size_t>
            class no_operation : public consumer<Input, InputTupletIdType>
            {
                using super = consumer<Input, InputTupletIdType>;

            public:
                using typename super::input_t;
                using typename super::input_tupletid_t;
                using typename super::input_chunk_t;

            protected:
                inline virtual void on_consume(const input_chunk_t *data) override final __attribute__((always_inline))
                {
                }
            };
        }
    }
}

