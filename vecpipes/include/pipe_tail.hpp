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

#include "vector.hpp"

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            template<class Input, class InputForwardIt = Input*>
            class consumer
            {
            public:
                using input_t = Input;
                using input_iterator_t = InputForwardIt;
                using input_vector_t = vector<input_t, input_iterator_t>;

                template<class IL, class IR, class ILF, class IRF>
                friend class bi_pipe_tail;

            protected:
                virtual void on_consume(input_iterator_t *begin, input_iterator_t *end) { };

                virtual input_t lookup(input_iterator_t *ptr) final
                {
                    return **ptr;
                }

                virtual input_iterator_t as_reference(input_iterator_t *ptr) final
                {
                    return *ptr;
                }


            public:
                virtual void close() { }

                virtual void consume(input_vector_t *data) final
                {
                    auto iterator = data->get_iterator();
                    if (!iterator.is_empty()) {
                        on_consume(iterator.begin, iterator.end);
                    }
                }

                virtual void consume(input_iterator_t *begin, input_iterator_t *end) final
                {
                    assert (begin != nullptr);
                    assert (end != nullptr);
                    assert (begin <= end);
                    on_consume(begin, end);
                }
            };

        }
    }
}