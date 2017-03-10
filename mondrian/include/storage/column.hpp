#pragma once

#include <vpipes.hpp>
#include <cstddef>
#include <cassert>
#include <stdexcept>
#include <cstring>
#include <memory>
#include <vpipes.hpp>

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

        private:
            value_t *data;
            size_t capacity, size;

            template <class V>
            class invoke_filter : public vpipes::producer<V>
            {
                using super = vpipes::producer<V>;
                using typename super::consumer_t;

                const V *begin, *end;
            public:
                invoke_filter(consumer_t *consumer, const V *begin, const V *end, unsigned int chunk_size) :
                        super(consumer, chunk_size), begin(begin), end(end) {
                    assert (begin != nullptr && end != nullptr);
                    assert (begin <= end);
                }

                virtual void on_start() override
                {
                    size_t distance = (end - begin);
                    for (size_t i = 0; i != distance; ++i)
                        super::produce(&i);
                }
            };

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

            vpipes::producer<value_t> *table_scan(vpipes::consumer<value_t> *consumer,
                                                typename vpipes::functional::batched_predicate<value_t>::func_t predicate_func,
                                                unsigned chunk_size)
            {
                return new invoke_filter<value_t>(consumer, data, data + size, chunk_size);
            }

        private:

        };
    }
}
