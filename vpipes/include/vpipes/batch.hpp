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

#include "../vpipes.hpp"

namespace mondrian
{
    namespace vpipes
    {
        enum cpu_hint { for_read, for_write };

        template<class ValueType>
        class batch
        {
        public:
            using value_t = ValueType;
            using block_copy_t = typename block_copy<value_t>::func_t;
            using block_null_copy_t = typename block_null_copy::func_t;

        private:
            mtl::smart_array<tuplet_id_t> tupletids;
            mtl::smart_array<value_t> values;
            mtl::smart_bitmask null_mask;

            size_t max_size, cursor;

        public:
            enum class state
            {
                full, non_full
            };

            batch(__in__ size_t num_of_elements) :
                        max_size(num_of_elements), cursor(0),
                        tupletids(num_of_elements),
                        values(num_of_elements),
                        null_mask(num_of_elements)
            {
                null_mask.resize(num_of_elements);
            }

            inline void reset()
            {
                cursor = 0;
                null_mask.reset();
            }

            inline void prefetch(cpu_hint hint)
            {
                if (hint == cpu_hint::for_read) {
                    tupletids.prefetch(mtl::cpu_hint::for_read);
                    values.prefetch(mtl::cpu_hint::for_read);
                    null_mask.prefetch(mtl::cpu_hint::for_read);
                } else {
                    tupletids.prefetch(mtl::cpu_hint::for_write, cursor);
                    values.prefetch(mtl::cpu_hint::for_write, cursor);
                    null_mask.prefetch(mtl::cpu_hint::for_write, cursor);
                }
            }

            inline void iota(__in__ tuplet_id_t start,
                             __in__ size_t num_of_values,
                             __in__ block_copy_t block_copy_func,
                             __in__ block_null_copy_t block_null_copy_func) __attribute__((always_inline))
            {
                assert (num_of_values <= max_size);
                num_of_values = MIN(max_size, num_of_values);
                auto end = start + num_of_values;

                tupletids.iota(cursor, num_of_values, start);
                block_copy_func(values.get_raw_data(), start, end);

                null_mask.reset();
                null_mask.resize(num_of_values);

                auto null_mask_pos_start = 0;
                auto null_mask_pos_end = (num_of_values % (max_size + 1));
                assert (null_mask_pos_start < null_mask_pos_end);
                mtl::smart_array<size_t> null_mask_indices(null_mask_pos_end);
                null_mask_indices.iota(null_mask_pos_start, null_mask_pos_end, null_mask_pos_start);

                block_null_copy_func(&null_mask, &null_mask_indices, &tupletids);
                assert (null_mask.get_num_elements() == num_of_values);
                //printf("PRODUCER '%s' created batch:\t\t\t\t\t\t\t\t\t\t", creator_class_name);
                //null_mask.to_string(stdout);
                //printf("\n");

                null_mask_indices.dispose();

                cursor += num_of_values;
            }

            inline size_t add(__out__ state *out_state,
                              __in__ const tuplet_id_t *in_tuplet_ids,
                              __in__ const value_t *in_values,
                              __in__ const mtl::smart_bitmask *in_null_mask,
                              __in__ const size_t *indices,
                              __in__ size_t num_indices)
                              __attribute__((always_inline))
            {
                assert (cursor + 1 <= max_size);
                assert (out_state != nullptr && in_tuplet_ids != nullptr && in_values != nullptr && in_null_mask != nullptr);

                auto append_max_len = std::min(max_size - cursor, num_indices);
                auto retval = num_indices - append_max_len;

                values.gather_unsafe(indices, append_max_len, in_values, cursor);
                tupletids.gather_unsafe(indices, append_max_len, in_tuplet_ids, cursor);
                null_mask.override_by(0, in_null_mask, append_max_len);

                cursor += append_max_len;
                *out_state = (cursor >= max_size ? state::full : state::non_full);
                return retval;
            }

            inline size_t add(__out__ state *out_state,
                              __in__ const tuplet_id_t *in_tuplet_ids,
                              __in__ const value_t *in_values,
                              __in__ const mtl::smart_bitmask *in_null_mask,
                              __in__ size_t num_elements) __attribute__((always_inline))
            {
                assert (cursor + 1 <= max_size);
                auto append_max_len = std::min(max_size - cursor, num_elements);
                auto retval = num_elements - append_max_len;
                tupletids.set(cursor, in_tuplet_ids, append_max_len);
                values.set(cursor, in_values, append_max_len);
                null_mask.override_by(cursor, in_null_mask, append_max_len);
                cursor += append_max_len;
                *out_state = (cursor >= max_size ? state::full : state::non_full);
                return retval;
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
                tupletids.dispose();
                values.dispose();
                null_mask.dispose();
            }

            const value_t *get_values() const
            {
                return values.get_content();
            }

            const tuplet_id_t *get_tupletids() const
            {
                return tupletids.get_content();
            }

            const mtl::smart_bitmask *get_null_mask() const
            {
                return &null_mask;
            }

            null_info get_null_info() const
            {
                auto info = null_mask.get_info();
                return (info == mtl::bitset_info::non_true ? null_info::non_null : null_info::contains_null);
            }
        };
    }
}

