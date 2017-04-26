// Mondrian Template Library - data structures, explict management and algorithm tailored to modern database systems
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

#include "../../mtl"

#define IDX_TO_BLOCK(idx,bitmask_ptr)                                                   \
    size_t(std::ceil(idx / (sizeof(bitmask_ptr->get_content()->get_value_size()) << 2)))

#define IDX_TO_BIT(idx,block)                               \
    idx - (block << 5);

#define PREPARE_BIT_SET(idx, value)                         \
    set_flag |= value;                                      \
    idx += offset;                                          \
    auto block_id = IDX_TO_BLOCK(idx, this);                \
    auto bit_id = IDX_TO_BIT(idx, block_id);                \

#define REBUILD_SET_FLAG()                                      \
    {                                                           \
        set_flag = false;                                       \
        auto block_idx = content.get_num_elements();            \
        auto content_data = content.get_raw_data();             \
        while (block_idx--) {                                   \
            set_flag |= *content_data++;                        \
        }                                                       \
    };

#define SMART_BITMASK_GET_UNSAFE_FAST(out_bool_is_set, bit_mask, idx)                           \
    {                                                                                           \
        auto __idx = idx;                                                                       \
        __idx += bit_mask->get_offset();                                                        \
        auto block_id = IDX_TO_BLOCK(__idx, bit_mask);                                          \
        auto bit_id = IDX_TO_BIT(__idx, block_id);                                              \
        auto mask = (1 << bit_id);                                                              \
        auto modable_raw = const_cast<uint32_t *>(bit_mask->get_content()->get_raw_data());     \
        out_bool_is_set = ((modable_raw[block_id] & mask) == mask);                             \
    }


namespace mondrian
{
    namespace mtl
    {
        class smart_bitmask
        {
            using smart_array_t = smart_array<uint32_t>;
        private:
            smart_array_t content;
            size_t offset, max_idx;
            bool set_flag;

        public:

            smart_bitmask(size_t initial_capacity = 16, float grow_factor = 1.5f):
                    content(std::ceil(initial_capacity / float(8 * sizeof(content.get_value_size()))), grow_factor,
                            init_value_policy::zero_memory), offset(0), max_idx(0), set_flag(false) { }

            smart_bitmask(const smart_bitmask &other) = delete;
            smart_bitmask(smart_bitmask && other) = delete;

            virtual inline void set_safe(size_t idx, bool value) final __attribute__((always_inline))
            {
                PREPARE_BIT_SET(idx, value);
                auto block = *content.get_safe(block_id);
                block |= (value << bit_id);
                content.set(block_id, block);
                max_idx = std::max(max_idx, idx);
                assert (get_unsafe(idx - offset) == value);
            }

            virtual inline void set_unsafe(size_t idx, bool value) final __attribute__((always_inline))
            {
                PREPARE_BIT_SET(idx, value);
                auto block = content.get_unsafe(block_id);
                *block |= (value << bit_id);
                max_idx = std::max(max_idx, idx);
                assert (get_unsafe(idx - offset) == value);
            }

            virtual inline void override_by(size_t idx, const smart_bitmask *bits, size_t num_values) final __attribute__((always_inline))
            {
                idx += offset;
                while (num_values--) {
                    bool value = bits->get_unsafe(idx);
                    set_flag |= value;
                    set_unsafe(idx, value);
                    idx++;
                }
                max_idx = std::max(max_idx, idx);
            }

            virtual inline void gather_unsafe(const size_t *indices, size_t num_indices, const smart_bitmask *src,
                                              size_t this_start_idx = 0) final __attribute__((always_inline))
            {
                while (num_indices--) {
                    bool value = src->get_unsafe(*indices++);
                    set_flag |= value;
                    set_unsafe(this_start_idx++, value);
                }
            }

            virtual inline const uint32_t* get_raw_data() const final __attribute__((always_inline))
            {
                return content.get_raw_data();
            }

            virtual inline const smart_array<uint32_t> *get_content() const final __attribute__((always_inline))
            {
                return &content;
            }

            virtual inline bool get_unsafe(size_t idx) const final __attribute__((always_inline))
            {
                idx += offset;
                auto block_id = IDX_TO_BLOCK(idx, this);
                auto bit_id = IDX_TO_BIT(idx, block_id);
                auto mask = (1 << bit_id);
                bool return_value = ((content.get_raw_data()[block_id] & mask) == mask);
                return return_value;
            }

            virtual inline bool get_safe(size_t idx) final __attribute__((always_inline))
            {
                idx += offset;
                auto block_id = IDX_TO_BLOCK(idx, this);
                auto bit_id = IDX_TO_BIT(idx, block_id);
                auto mask = (1 << bit_id);
                return ((*content.get_safe(block_id) & mask) == mask);
            }

            virtual inline size_t get_num_elements() const final __attribute__((always_inline))
            {
                return max_idx - offset + 1;
            }

            virtual inline void prefetch(cpu_hint hint, size_t idx = 0) final __attribute__((always_inline))
            {
                idx += offset;
                auto block_id = IDX_TO_BLOCK(idx, this);
                if (hint == cpu_hint::for_read) {
                    __builtin_prefetch(content.get_raw_data() + block_id, 0, 0);
                } else {
                    __builtin_prefetch(content.get_raw_data() + block_id, 1, 0);
                }
            }

            virtual inline void dispose() final __attribute__((always_inline))
            {
                content.dispose();
            }

            bitset_info get_info() const
            {
                auto num_elems = content.get_num_elements();
                auto base = content.get_raw_data();
                while (num_elems--) {
                    if ((*base++) > 0) {
                        return bitset_info::contains_true;
                    }
                }
                return bitset_info::non_true;
            }

            virtual void unset_all() final __attribute__((always_inline))
            {
                content.set_all(0);
                set_flag = false;
            }

            virtual void set_all() final __attribute__((always_inline))
            {
                content.set_all(std::numeric_limits<smart_array<uint32_t>::type_t>::max());
                set_flag = true;
            }

            virtual void unset_range_unsafe(size_t begin, size_t end) final __attribute__((always_inline))
            {
                for (auto idx = begin; idx != end; ++idx)
                    this->set_unsafe(idx, false);
                REBUILD_SET_FLAG();
            }

            virtual void unset_range_safe(size_t begin, size_t end) final __attribute__((always_inline))
            {
                for (auto idx = begin; idx != end; ++idx)
                    this->set_safe(idx, false);
                REBUILD_SET_FLAG();
            }

            virtual void reset() final __attribute__((always_inline))
            {
                unset_all();
                offset = max_idx = 0;
            }

            virtual void set_offset(size_t idx) final __attribute__((always_inline))
            {
                offset = idx;
            }

            virtual size_t get_offset() const final __attribute__((always_inline))
            {
                return offset;
            }

            virtual void resize(size_t num_elements) final __attribute__((always_inline))
            {
                content.resize(IDX_TO_BLOCK(num_elements, this) + 1);
                max_idx = num_elements - 1;
                unset_all();
            }

            virtual bool is_unset() const final __attribute__((always_inline))
            {
                return !set_flag;
            }

            void to_string(FILE *file) const
            {
                std::string mask;
                for (size_t i = 0; i < content.get_num_elements(); ++i) {
                    auto value = *content.get_unsafe(i);
                    std::bitset<sizeof(smart_array_t::type_t)*8> x(value);
                    mask.append(x.to_string());
                }

                fprintf(file, "smart_bitmask(offset=%zu, max_idx=%zu, num_elements=%zu, set_flag=%d, size_in_byte=%zu, mask='%s'",
                        offset, max_idx, this->get_num_elements(), set_flag, content.get_value_size() * content.get_num_elements(),
                        mask.c_str());
            }

        };
    }
}