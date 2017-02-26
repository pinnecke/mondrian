#pragma once

#include <query_engine/operators/vector.hpp>

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            template<class Input, class InputForwardIt = Input*>
            class pipe_tail
            {
            public:
                using input_t = Input;
                using input_iterator_t = InputForwardIt;
                using input_vector_t = vector<input_t, input_iterator_t>;

            protected:
                virtual void on_consume(const input_iterator_t *begin, const input_iterator_t *end) { };

            public:
                virtual void close() { };

                virtual void consume(const input_vector_t *data) final
                {
                    auto iterator = data->get_iterator();
                    if (!iterator.is_empty())
                        on_consume(iterator.begin, iterator.end);
                }
            };

        }
    }
}