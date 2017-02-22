#pragma once

#include <cassert>
#include <query_engine/operators/iterator.hpp>

namespace mondrian {
    namespace query_engine {
        namespace operators {

            template<class ValueType>
            class vector {
                const ValueType **data;
                size_t max_size, cursor;
            public:
                enum class state {
                    full, non_full
                };

                vector(size_t num_of_elements) : max_size(num_of_elements), cursor(0) {
                    data = (const ValueType **) malloc(this->max_size * sizeof(ValueType *));
                }

                state add(const ValueType *value) {
                    assert(cursor < max_size);
                    data[cursor++] = value;
                    return (cursor == max_size ? state::full : state::non_full);
                }

                iterator <ValueType> get_iterator() const {
                    return iterator<ValueType>(data, data + cursor);
                }

                void release() { free(data); }
            };

        }
    }
}

