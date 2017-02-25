#pragma once

#include <query_engine/operators/vector.hpp>

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            template<class InputLeft, class InputRight,
                    class InputLeftForwardIt = InputLeft*, class InputRightForwardIt = InputRight*>
            class bi_consumer
            {
            public:
                using input_left_t = InputLeft;
                using input_left_iterator_t = InputLeftForwardIt;
                using input_left_vector_t = vector<input_left_t, input_left_iterator_t>;

                using input_right_t = InputRight;
                using input_right_iterator_t = InputRightForwardIt;
                using input_right_vector_t = vector<input_right_t, input_right_iterator_t>;

                virtual void consume_left(const input_left_vector_t *data) = 0;
                virtual void consume_right(const input_right_vector_t *data) = 0;
                virtual void close() { };
            };

        }
    }
}