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
#include <numeric>
#include "iterator.hpp"
#include "macros.hpp"

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

            inline void reset()
            {
                cursor = 0;
            }

            inline void memory_prefetch_for_read()
            {
                __builtin_prefetch(data, PREFETCH_RW_FOR_READ, PREFETCH_LOCALITY_KEEP_IN_CACHES_HIGH);
            }

            inline void memory_prefetch_for_write()
            {
                __builtin_prefetch(data + cursor, PREFETCH_RW_FOR_WRITE, PREFETCH_LOCALITY_KEEP_IN_CACHES_HIGH);
            }

            inline state add(tupletid_t value)
            {
                assert(cursor < max_size);
                data[cursor++] = value;
                return (cursor == max_size ? state::full : state::non_full);
            }

            inline void iota(tupletid_t start, size_t num_of_values) __attribute__((always_inline))
            {
                assert (num_of_values <= max_size);
                num_of_values = MIN(max_size, num_of_values);
                std::iota(data, data + num_of_values, start);
                cursor += num_of_values;
            }

            inline tupletid_t *add(state *out, tupletid_t *begin, tupletid_t *end) __attribute__((always_inline))
            {
                assert (cursor + 1 <= max_size);
                auto append_max_len = std::min(max_size - cursor, size_t(end - begin));
                for (auto it = begin; it != begin + append_max_len; ++it) {
                    data[cursor++] = *it;
                }
                *out = (cursor == max_size ? state::full : state::non_full);
                return begin + append_max_len;
            }

            inline iterator <value_t> get_iterator()
            {
                return iterator<value_t>(data, data + cursor);
            }

            void release() { free(data); }
        };
    }
}

