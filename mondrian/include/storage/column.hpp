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
            using predicate_t = typename vpipes::functional::batched_predicate<value_t>::func_t;

        private:
            value_t *data;
            size_t capacity, size;

            class invoke_filter : public vpipes::producer<value_t>
            {
                using super = vpipes::producer<value_t>;
                using typename super::consumer_t;

                const value_t *begin, *end;
            public:
                invoke_filter(consumer_t *consumer, const value_t *begin, const value_t *end, unsigned int chunk_size) :
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
                                                  predicate_t predicate,
                                                  unsigned chunk_size)
            {
                auto filter = new vpipes::toolkit::batched_pred_filter<value_t>(consumer,
                                                                                [&] (value_t *out_begin, value_t *out_end,
                                                                                         const size_t *begin, const size_t*end)
                                                                                {
                                                                                    assert (out_end - out_begin >= end - begin);
                                                                                    size_t distance = (end - begin);
                                                                                    for (size_t i = 0; i != distance; ++i) {
                                                                                        out_begin[i] = data[begin[i]];
                                                                                    }
                                                                                },
                                                                                predicate, chunk_size);
                return new invoke_filter(filter, data, data + size, chunk_size);
            }

        private:

        };
    }
}
