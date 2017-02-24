#pragma once

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {

            template<class ValueType, class ValuePointerType = ValueType*>
            struct iterator
            {
                using value_t = ValueType;
                using value_pointer_t = ValuePointerType;

                const value_pointer_t *begin;
                const value_pointer_t *end;

                iterator(const value_pointer_t *begin, const value_pointer_t *end) : begin(begin), end(end) {}

                bool is_empty() { return begin == end; }
            };

        }
    }
}

