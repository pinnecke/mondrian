#pragma once

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
#include "../pipe.hpp"

using namespace std;

namespace mondrian
{
    namespace vpipes
    {
        namespace toolkit
        {
            template <class InputType, class InputTupletIdType = size_t>
            class table_scan : public producer<InputType, InputTupletIdType>
            {
                using super = producer<InputType, InputTupletIdType>;

            public:
                using input_t = InputType;
                using input_tupletid_t = InputTupletIdType;
                using typename super::consumer_t;
                using filter_t = filter<input_t>;
                using predicate_t = typename filter_t::predicate_t;
                using materializer_t = typename functional::batched_materializes<input_t, input_tupletid_t>::func_t;

            private:
                const input_t *begin, *end;
                filter_t *filter;
                materializer_t materialize_func;

            public:
                table_scan(consumer_t *consumer, const input_t *begin, const input_t *end,
                           predicate_t predicate, materializer_t materializer, unsigned int chunk_size) :
                        super(nullptr, chunk_size), begin(begin), end(end), materialize_func(materializer)
                {
                    assert (begin != nullptr && end != nullptr);
                    assert (begin <= end);
                    filter = new filter_t(consumer, materialize_func, predicate, chunk_size);
                    super::set_consumer(filter);
                }

                virtual void on_start() override
                {
                    assert (filter != nullptr);
                    super::produce_tupletid_range(0, (end - begin));
                }

                vortial void on_close() override
                {
                    assert (filter != nullptr);
                    filter->close();
                }

                virtual void on_cleanup() override
                {
                    assert (filter != nullptr);
                    delete filter;
                    filter = nullptr;
                }
            };
        }
    }
}