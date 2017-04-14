#pragma once

#include <vpipes.hpp>
#include <cstddef>
#include <cassert>
#include <stdexcept>
#include <cstring>
#include <memory>
#include <vpipes.hpp>
#include "utils/gather.hpp"

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
            using table_scan_t = pipes::table_scan<ValueType>;
            using predicate_func_t = typename table_scan_t::predicate_func_t;
            using point_copy_t = typename point_copy<value_t, size_t>::func_t;
            using point_null_copy_t = typename point_null_copy<size_t>::func_t;

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

            inline virtual void materialize(value_t *out, const tupletid_t *tupletids,
                                            size_t num_of_ids) final __attribute__((always_inline))
            {
                assert (out != nullptr && tupletids != nullptr);
                while (num_of_ids--) {
                    *out++ = data[*tupletids++];
                }
            }

            point_copy_t f = [&] (value_t *out, const tupletid_t *tupletids, size_t num_of_ids) {
                this->materialize(out, tupletids, num_of_ids);
            };

            point_null_copy_t null_mask_f = [&] (mtl::smart_bitmask *out, const tupletid_t *tupletids, size_t num_of_ids) {
                //for (size_t idx = 0; idx < num_of_ids; ++idx) {
                //    out->set(idx, false); // TODO: For Testing
                //}
            };

            inline virtual producer<value_t> *table_scan(consumer<value_t> *consumer, predicate_func_t predicate,
                                                         unsigned scan_batch_size, unsigned filter_batch_size,
                                                         bool filter_hint_expected_avg_batch_eval_is_non_empty) final __attribute__((always_inline))
            {
                size_t start = 0, end = size;
                interval<size_t> all_tuplet_ids(start, end);
                return new pipes::table_scan<value_t>(consumer, &all_tuplet_ids, &all_tuplet_ids + 1, predicate,
                                                        [&] (value_t *out, tupletid_t begin, tupletid_t end)
                                                        {
                                                            assert (out != nullptr);
                                                            assert (begin < end);
                                                            memcpy(out, data + begin, (end - begin) * sizeof(value_t));
                                                        },
                                                        [&] (mtl::smart_bitmask *out, tupletid_t begin, tupletid_t end)
                                                        {
                                                            assert (out != nullptr);
                                                            assert (begin < end);
                                                            out->unset_range_safe(0, (end - begin));
                                                        },
                                                        scan_batch_size, filter_batch_size,
                                                        filter_hint_expected_avg_batch_eval_is_non_empty);
            }
        };
    }
}
