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

namespace mondrian
{
    namespace mtl
    {
        template<class Type>
        class smart_array
        {
            using self = smart_array<Type>;
        public:
            using type_t = Type;

            class assignment_proxy
            {
                friend class smart_array;
                type_t *ref;

            public:
                assignment_proxy &operator=(const type_t &value)
                {
                    *ref = value;
                    return *this;
                }

                type_t *get_address() const
                {
                    return ref;
                }
            };

        private:
            type_t *content;
            size_t size, capacity;
            float grow_factor;
            bool disposed;
            init_value_policy policy;

            assignment_proxy proxy;

        public:
            smart_array(size_t initial_capacity, float grow_factor = 1.5f,
                        init_value_policy policy = init_value_policy::dont_care):
                    capacity(initial_capacity), grow_factor(grow_factor), disposed(false), size(0), policy(policy)
            {
                assert (capacity > 0);
                assert (grow_factor > 0);
                content = (type_t *) malloc (capacity * sizeof(type_t));
                assert (content != nullptr);
                if (policy == init_value_policy::zero_memory) {
                    memset(content, 0, capacity * sizeof(type_t));
                }
            }

            smart_array() = delete;
            smart_array(const self &other) = delete;
            smart_array(self && other) = delete;

            virtual inline void auto_resize(size_t required_size) final __attribute__((always_inline))
            {
                if (__builtin_expect((required_size > capacity), false)) {
                    auto old_capacity = capacity;
                    while (required_size > capacity)
                        capacity = std::ceil(capacity * grow_factor);
                    content = (type_t *) realloc (content, capacity * sizeof(type_t));
                    assert (content != nullptr);
                    if (policy == init_value_policy::zero_memory) {
                        memset(content + old_capacity, 0, (capacity - old_capacity) * sizeof(type_t));
                    }
                }
            }

            constexpr size_t get_value_size() const
            {
                return sizeof(type_t);
            }

            virtual inline void set(size_t idx, const type_t &value) final __attribute__((always_inline))
            {
                auto upper = idx >= size ? idx + 1 : size;
                auto_resize (upper);
                size = upper;
                assert (idx < size);
                set_unsafe(idx, value);
            }

            virtual inline void set_unsafe(size_t idx, const type_t &value) final __attribute__((always_inline))
            {
                content[idx] = value;
            }

            virtual inline void set_unsafe(size_t idx, const type_t *value) final __attribute__((always_inline))
            {
                content[idx] = *value;
            }

            virtual inline void set(size_t idx, const type_t *data, size_t num_values) final __attribute__((always_inline))
            {
                auto upper = idx + num_values;
                assert (data != nullptr);
                auto_resize(upper);
                memcpy(content + idx, data, num_values * sizeof(type_t));
                size = upper > size ? upper : size;
                assert (size <= capacity);
            }

            virtual inline void add(const type_t &element) final __attribute__((always_inline))
            {
                auto_resize (size + 1);
                content[size++] = element;
            }

            virtual inline type_t *get_raw_data() const final __attribute__((always_inline))
            {
                return content;
            }

            virtual inline size_t get_num_elements() const final __attribute__((always_inline))
            {
                return size;
            }

            virtual inline void add_all(const type_t *source, size_t num_elements) final __attribute__((always_inline))
            {
                auto_resize (size + num_elements);
                memcpy(this->content, source, num_elements * sizeof(type_t));
                size += num_elements;
            }

            virtual inline void dispose() final __attribute__((always_inline))
            {
                if (not disposed) {
                    free (content);
                    content = nullptr;
                    disposed = true;
                }
            }

            virtual inline bool is_disposed() final __attribute__((always_inline))
            {
                return disposed;
            }

            virtual inline const type_t *get_content() const final __attribute__((always_inline))
            {
                return content;
            }

            virtual inline void iota(size_t start_idx, size_t num_values, size_t inital_value) final __attribute__((always_inline))
            {
                auto base = content + start_idx;
                std::iota(base, base + num_values, inital_value);
            }

            virtual inline void prefetch(cpu_hint hint, size_t idx = 0) final __attribute__((always_inline))
            {
                if (hint == cpu_hint::for_read) {
                    __builtin_prefetch(content + idx, 0, 0);
                } else {
                    __builtin_prefetch(content + idx, 1, 0);
                }
            }

            virtual inline const type_t *get_safe(size_t idx) final __attribute__((always_inline))
            {
                auto idx_size = idx + 1;
                auto_resize (idx_size);
                size = (idx_size > size) ? idx_size : size;
                assert (size <= capacity);
                assert (idx < capacity);
                assert (idx < size);
                return get_unsafe(idx);
            }

            virtual inline type_t *get_unsafe(size_t idx) const final __attribute__((always_inline))
            {
                return content + idx;
            }

            virtual inline void set_all(const type_t &value) final __attribute__((always_inline))
            {
                memset (content, value, size * sizeof(type_t));
            }

            virtual inline void gather_unsafe(const size_t *indices, size_t num_indices, const smart_array *src,
                                              size_t this_start_idx = 0) final __attribute__((always_inline))
            {
                while (num_indices--) {
                    set_unsafe(this_start_idx++, src->get_unsafe(*indices++));
                }
            }

            virtual inline void gather_unsafe(const size_t *indices, size_t num_indices, const type_t *src,
                                              size_t this_start_idx = 0) final __attribute__((always_inline))
            {
                while (num_indices--) {
                    set_unsafe(this_start_idx++, src + (*indices++));
                }
            }

            virtual void resize(size_t num_elements) final __attribute__((always_inline))
            {
                auto_resize(num_elements);
                size = num_elements;
            }
        };
    }
}