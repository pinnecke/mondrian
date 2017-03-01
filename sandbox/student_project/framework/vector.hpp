#pragma once

#include <cassert>
#include "iterator.hpp"

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            template<class ValueType, class ValueForwardIt>
            class vector
            {
            public:
                using value_t = ValueType;
                using value_iterator_t = ValueForwardIt;

            private:
                value_iterator_t *data;
                size_t max_size, cursor;

            public:
                enum class state
                {
                    full, non_full
                };

                vector(size_t num_of_elements) : max_size(num_of_elements), cursor(0)
                {
                    data = (value_iterator_t *) malloc(this->max_size * sizeof(value_iterator_t));
                }

                state add(value_iterator_t value)
                {
                    assert(cursor < max_size);
                    data[cursor++] = value;
                    return (cursor == max_size ? state::full : state::non_full);
                }

                value_iterator_t *add(state *out, value_iterator_t *begin, value_iterator_t *end)
                {
                    auto append_max_len = std::min(max_size - cursor, size_t(end - begin));
                    for (auto it = begin; it != begin + append_max_len; ++it) {
                        data[cursor++] = *it;
                    }
                    *out = (cursor == max_size ? state::full : state::non_full);
                    return begin + append_max_len;
                }

                iterator <value_t> get_iterator()
                {
                    return iterator<value_t>(data, data + cursor);
                }

                void release() { free(data); }
            };

        }
    }
}

