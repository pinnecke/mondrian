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
        enum cpu_hint { for_read, for_write };

        template<class ValueType, class TupletIdType = size_t>
        class batch
        {
        public:
            using value_t = ValueType;
            using tupletid_t = TupletIdType;
            using block_copy_t = typename block_copy<value_t, tupletid_t>::func_t;

        private:
            tupletid_t *tupletids;
            value_t *values;
            size_t max_size, cursor;

            void allocate_buffers()
            {
                tupletids = (tupletid_t *) malloc(this->max_size * sizeof(tupletid_t));
                values = (value_t *) malloc(this->max_size * sizeof(value_t));
            }

            void copy_buffers(const tupletid_t *src_tupletids, const value_t *src_values)
            {
                memcpy(tupletids, src_tupletids, cursor * sizeof(tupletid_t));
                memcpy(values, src_values, cursor * sizeof(value_t));
            }

        public:
            enum class state
            {
                full, non_full
            };

            batch(size_t num_of_elements) : max_size(num_of_elements), cursor(0)
            {
                allocate_buffers();
            }

            batch(const batch<value_t, tupletid_t> *other)
            {
                max_size = other->max_size;
                cursor = other->cursor;
                allocate_buffers();
                copy_buffers(other->tupletids, other->values);
            }

            inline void reset()
            {
                cursor = 0;
            }

            inline void prefetch(cpu_hint hint)
            {
                if (hint == cpu_hint::for_read) {
                    __builtin_prefetch(tupletids, PREFETCH_RW_FOR_READ, PREFETCH_LOCALITY_KEEP_IN_CACHES_HIGH);
                    __builtin_prefetch(values, PREFETCH_RW_FOR_READ, PREFETCH_LOCALITY_KEEP_IN_CACHES_HIGH);
                } else {
                    __builtin_prefetch(tupletids + cursor, PREFETCH_RW_FOR_WRITE, PREFETCH_LOCALITY_KEEP_IN_CACHES_HIGH);
                    __builtin_prefetch(values + cursor, PREFETCH_RW_FOR_WRITE, PREFETCH_LOCALITY_KEEP_IN_CACHES_HIGH);
                }
            }

            inline void iota(tupletid_t start, size_t num_of_values, block_copy_t block_copy_func) __attribute__((always_inline))
            {
                assert (num_of_values <= max_size);
                num_of_values = MIN(max_size, num_of_values);
                auto offset = tupletids + cursor;
//               std::iota(offset, offset + num_of_values, start);
		 std::fill(offset, offset + num_of_values, 666);	
                block_copy_func(values, start, start + num_of_values);
                cursor += num_of_values;
            }

            inline size_t add(state *out, const tupletid_t *in_tuplet_ids, const value_t *in_values,
                                   const size_t *indices, size_t num_indices) __attribute__((always_inline))
            {
                assert (cursor + 1 <= max_size);
                auto append_max_len = std::min(max_size - cursor, num_indices);
                if (append_max_len > 10){
                    printf("");
                }
                auto retval = num_indices - append_max_len;
                while (append_max_len--) {
                    auto idx = *indices++;
                    *(values + cursor) = in_values[idx];
                    *(tupletids + cursor++) = in_tuplet_ids[idx];
                }
                *out = (cursor >= max_size ? state::full : state::non_full);
                return retval;
            }

            inline size_t add(state *out, const tupletid_t *in_tuplet_ids, const value_t *in_values,
                              size_t num_elements) __attribute__((always_inline))
            {
                assert (cursor + 1 <= max_size);
                auto append_max_len = std::min(max_size - cursor, num_elements);
                auto retval = num_elements - append_max_len;
                memcpy(tupletids + cursor, in_tuplet_ids, append_max_len * sizeof(tupletid_t));
                memcpy(values + cursor, in_values, append_max_len * sizeof(value_t));
                cursor += append_max_len;
                *out = (cursor >= max_size ? state::full : state::non_full);
                return retval;
            }

            inline iterator <value_t> get_iterator()
            {
                return iterator<value_t>(tupletids, tupletids + cursor);
            }

            bool is_empty() const
            {
                return cursor == 0;
            }

            size_t get_size() const
            {
                assert (cursor <= max_size);
                return cursor;
            }

            void release() {
                free(tupletids);
                free(values);
            }

            value_t *get_values_begin() const
            {
                return values;
            }

            value_t *get_values_end() const
            {
                return values + cursor;
            }

            tupletid_t *get_tupletids_begin() const
            {
                return tupletids;
            }

            tupletid_t *get_tupletids_end() const
            {
                return tupletids + cursor;
            }
        };
    }
}

