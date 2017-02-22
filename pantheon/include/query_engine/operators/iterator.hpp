#pragma once

namespace mondrian {
    namespace query_engine {
        namespace operators {

            template<class ValueType>
            struct iterator {
                const ValueType **begin;
                const ValueType **end;

                iterator(const ValueType **begin, const ValueType **end) : begin(begin), end(end) {}
            };

        }
    }
}

