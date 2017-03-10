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

#include <cassert>
#include <algorithm>
#include "iterator.hpp"

namespace mondrian
{
    namespace vpipes
    {
        template<class ValueType, class TupletIdType = size_t>
        class chunk
        {
        public:
            using value_t = ValueType;
            using tupletid_t = TupletIdType;

        private:
            tupletid_t *data;
            size_t max_size, cursor;

        public:
            enum class state
            {
                full, non_full
            };

            chunk(size_t num_of_elements) : max_size(num_of_elements), cursor(0)
            {
                data = (tupletid_t *) malloc(this->max_size * sizeof(tupletid_t));
            }

            state add(tupletid_t value)
            {
                assert(cursor < max_size);
                data[cursor++] = value;
                return (cursor == max_size ? state::full : state::non_full);
            }

            tupletid_t *add(state *out, tupletid_t *begin, tupletid_t *end)
            {
                assert (cursor + 1 <= max_size);
                auto append_max_len = std::min(max_size - cursor, size_t(end - begin));
                for (auto it = begin; it != begin + append_max_len; ++it) {
                    data[cursor++] = *it;
                }
                *out = (cursor == max_size ? state::full : state::non_full);
                return begin + append_max_len;
            }

            iterator <value_t> get_iterator()
            {
                return iterator<value_t>(data, data + cursor);
            }

            void release() { free(data); }
        };
    }
}

