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

#define IDX_TO_BLOCK(idx)                                                   \
    size_t(std::ceil(idx / (sizeof(content.get_value_size()) << 3)))

#define IDX_TO_BIT(idx,block)                               \
    idx - (block << 6);

namespace mondrian
{
    namespace mtl
    {
        class smart_bitmask
        {
        public:
            class assignment_proxy
            {
                friend class smart_bitmask;
                size_t idx;
                smart_bitmask *parent;

            public:
                assignment_proxy(smart_bitmask *parent): parent(parent) { }

                assignment_proxy &operator=(bool value)
                {
                    parent->set(idx, value);
                    return *this;
                }

                virtual inline bool operator==(bool value) final __attribute__((always_inline))
                {
                    return (parent->get(idx) == value);
                }

                virtual inline bool operator!=(bool value) final __attribute__((always_inline))
                {
                    return !(this->operator==(value));
                }

                virtual inline bool operator!() final __attribute__((always_inline))
                {
                    return !(parent->get(idx));
                }

            };

        private:
            smart_array<uint64_t> content;
            assignment_proxy proxy;

        public:

            smart_bitmask(size_t initial_capacity, float grow_factor = 1.5f):
                    content(std::ceil(initial_capacity / float(8 * sizeof(content.get_value_size()))), grow_factor,
                            init_value_policy::zero_memory), proxy(this) { }

            smart_bitmask() = delete;
            smart_bitmask(const smart_bitmask &other) = delete;
            smart_bitmask(smart_bitmask && other) = delete;

            virtual inline void set(size_t idx, bool value) final __attribute__((always_inline))
            {
                auto block_id = IDX_TO_BLOCK(idx);
                auto bit_id = IDX_TO_BIT(idx, block_id);
                auto block = content[block_id].get_address();
                *block |= (1 << bit_id);
            }

            virtual inline bool get(size_t idx) final __attribute__((always_inline))
            {
                auto block_id = IDX_TO_BLOCK(idx);
                auto bit_id = IDX_TO_BIT(idx, block_id);
                auto mask = (1 << bit_id);
                return (*content[block_id].get_address() & mask) == mask;
            }

            virtual inline assignment_proxy& operator[](size_t idx) final __attribute__((always_inline))
            {
                proxy.idx = idx;
                return proxy;
            }

            virtual inline void prefetch(cpu_hint hint, size_t idx = 0) final __attribute__((always_inline))
            {
                auto block_id = IDX_TO_BLOCK(idx);
                if (hint == cpu_hint::for_read) {
                    __builtin_prefetch(content.get_raw_data() + block_id, 0, 0);
                } else {
                    __builtin_prefetch(content.get_raw_data() + block_id, 1, 0);
                }
            }

        };
    }
}