#pragma once

#include <functional>
#include <query_engine/operators/pipes/sequential_filter.hpp>

using namespace std;

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            class query
            {
                unsigned vector_size;
            public:
                query(unsigned vector_size): vector_size(vector_size) { }

                template<class Type, class ForwardIt = Type*>
                query &filter(function<bool(const ForwardIt)> predicate)
                {
                    sql::sequential_filter<unsigned>(nullptr, vector_size, predicate);
                    // TODO: ...
                    return *this;
                }
            };
        }
    }
}