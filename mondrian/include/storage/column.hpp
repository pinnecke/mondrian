#pragma once

#include <vpipes.hpp>
#include <cstddef>
#include <cassert>
#include <stdexcept>
#include <cstring>
#include <memory>

namespace mondrian
{
    namespace storage
    {
        template <class ValueType>
        class column
        {
        public:
            using value_t = ValueType;

        private:
            value_t *data;
            size_t capacity, size;

        public:
            column(const value_t *begin, const value_t *end): capacity(end - begin), size(0)
            {
                assert (capacity > 0);
                assert (begin != nullptr && end != nullptr);
                assert (begin < end);
                if ((data = (value_t *) malloc (++capacity * sizeof(value_t))) == nullptr)
                    throw std::runtime_error("Malloc failed");
                assert (data != nullptr);
                append (begin, end);
            }

            column(size_t capacity): capacity(capacity), size(0)
            {
                assert (capacity > 0);
                if ((data = (value_t *) malloc (capacity * sizeof(value_t))) == nullptr)
                    throw std::runtime_error("Malloc failed");
                assert (data != nullptr);
            }

            void append(const value_t *begin, const value_t *end)
            {
                assert (begin != nullptr && end != nullptr);
                assert (begin < end);
                assert (data != nullptr);
                auto distance = (end - begin);
                auto append_pos = size;
                if (size + distance >= capacity) {
                    do capacity *= 1.4f; while (size + distance >= capacity);
                    if ((data = (value_t *) realloc(data, capacity * sizeof(value_t))) == nullptr)
                        throw std::runtime_error("Realloc failed");
                }
                memcpy(data + append_pos, begin, distance * sizeof(value_t));
                size += distance;
            }

            bool is_nullable()
            {
                return false;
            }

            vpipes::pipe_head<value_t > *table_scan(vpipes::consumer<value_t> *consumer, unsigned vector_size)
            {
                return new vpipes::toolkit::reader<value_t>(consumer, data, data + size, vector_size);
            }

        private:

        };
    }
}
