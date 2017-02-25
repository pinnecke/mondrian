#pragma once

#include <query_engine/operators/vector.hpp>

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            template<class Input, class InputForwardIt = Input*>
            class consumer
            {
            public:
                using input_t = Input;
                using input_iterator_t = InputForwardIt;
                using input_vector_t = vector<input_t, input_iterator_t>;

                virtual void consume(const input_vector_t *data) = 0;
                virtual void close() { };
            };

        }
    }
}