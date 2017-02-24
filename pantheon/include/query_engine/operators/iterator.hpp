#pragma once

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {

            template<class ValueType, class ValueForwardIt = ValueType*>
            struct iterator
            {
                using value_t = ValueType;
                using value_iterator_t = ValueForwardIt;

                const value_iterator_t *begin;
                const value_iterator_t *end;

                iterator(const value_iterator_t *begin, const value_iterator_t *end) : begin(begin), end(end) {}

                bool is_empty() { return begin == end; }
            };

        }
    }
}

