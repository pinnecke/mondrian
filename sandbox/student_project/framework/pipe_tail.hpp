#pragma once

#include "vector.hpp"

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

                template<class IL, class IR, class ILF, class IRF>
                friend class bi_pipe_tail;

            protected:
                virtual void on_consume(input_iterator_t *begin, input_iterator_t *end) { };

                virtual input_t lookup(input_iterator_t *ptr) final
                {
                    return **ptr;
                }

                virtual input_iterator_t as_reference(input_iterator_t *ptr) final
                {
                    return *ptr;
                }


            public:
                virtual void close() { }

                virtual void consume(input_vector_t *data) final
                {
                    auto iterator = data->get_iterator();
                    if (!iterator.is_empty()) {
                        on_consume(iterator.begin, iterator.end);
                    }
                }

                virtual void consume(input_iterator_t *begin, input_iterator_t *end) final
                {
                    assert (begin != nullptr);
                    assert (end != nullptr);
                    assert (begin <= end);
                    on_consume(begin, end);
                }
            };

        }
    }
}