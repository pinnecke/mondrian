#pragma once

#include <vpipes.hpp>
#include <cstddef>
#include <cassert>
#include <stdexcept>
#include <cstring>
#include <memory>
#include <vpipes.hpp>

using namespace mondrian::vpipes;

namespace mondrian
{
    namespace storage
    {
        template <class ValueType>
        class column
        {
        public:
            using value_t = ValueType;
            using tupletid_t = size_t;
            using table_scan_t = toolkit::table_scan<ValueType>;
            using predicate_t = typename table_scan_t::predicate_t;

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

            const value_t *materialize(tupletid_t tid)
            {
                assert(tid >= 0 && tid < size);
                return data + tid;
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

            producer<value_t> *table_scan(consumer<value_t> *consumer, predicate_t predicate, unsigned chunk_size)
            {
                interval<size_t> all_tuplet_ids(0, size);
                return new toolkit::table_scan<value_t>(consumer, &all_tuplet_ids, &all_tuplet_ids + 1, predicate,
                                                        [&] (value_t *out_begin, value_t *out_end,
                                                             const size_t *begin, const size_t*end)
                                                        {
                                                            assert (out_end - out_begin >= end - begin);
                                                            size_t distance = (end - begin);
                                                            for (size_t i = 0; i != distance; ++i) {
                                                                out_begin[i] = data[begin[i]];
                                                            }
                                                        }, chunk_size);
            }
        };
    }
}
