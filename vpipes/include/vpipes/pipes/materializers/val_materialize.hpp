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
            template<class Input>
            class val_materialize : public materialize<Input, Input>
            {
                using super = materialize<Input, Input>;

            public:
                using typename super::input_t;
                using typename super::input_batch_t;
                using typename super::destination_t;

            public:
                val_materialize(__out__ destination_t *destination,
                                __out__ size_t *result_set_size): super(destination, result_set_size) { }

            protected:
                inline virtual void invoke_memcpy(__out__ destination_t *destination,
                                                  __in__ const input_batch_t *data)
                override final __attribute__((always_inline))
                {
                    memcpy(destination, data->get_values(), data->get_size() * sizeof(destination_t));
                }

                virtual const char *get_class_name() const override
                {
                    return "vpipes::pipes::val_materialize";
                }
            };
        }
    }
}

