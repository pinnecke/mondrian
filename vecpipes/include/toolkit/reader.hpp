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

#include "../pipe_head.hpp"

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            namespace sources
            {
                template<class Output, class OutputForwardsIt = Output*>
                class reader : public pipe_head<Output, OutputForwardsIt>
                {
                    using super = pipe_head<Output, OutputForwardsIt>;

                public:
                    using typename super::input_t;
                    using typename super::input_iterator_t;
                    using typename super::consumer_t;

                    reader(consumer_t *consumer, input_iterator_t begin, input_iterator_t end,
                           unsigned vector_size): super(consumer, begin, end, vector_size) {}

                    virtual void on_start() override
                    {
                        auto begin = super::get_begin();
                        auto end = super::get_end();
                        for (auto it = begin; it != end; ++it)
                            super::produce(&it);
                        super::close();
                    };
                };

            }
        }
    }
}

