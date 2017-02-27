#pragma once

#include <query_engine/operators/vector.hpp>
#include <logger.hpp>

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

                template<class IL, class IR, class ILF, class IRF>
                friend class bi_pipe_tail;

            protected:
                virtual void on_consume(const input_iterator_t *begin, const input_iterator_t *end) { };

                virtual input_t lookup(const input_iterator_t *ptr) final
                {
                    return **ptr;
                }

                virtual const input_iterator_t as_reference(const input_iterator_t *ptr) final
                {
                    return *ptr;
                }


            public:
                virtual void close() { }

                virtual void consume(const input_vector_t *data) final
                {
                    auto iterator = data->get_iterator();
                    if (!iterator.is_empty()) {
                        LOG_INFO("Pipe tail %p receives vector: size %d", this, (iterator.end - iterator.begin));
                        on_consume(iterator.begin, iterator.end);
                    }
                }

                virtual void consume(const input_iterator_t *begin, const input_iterator_t *end) final
                {
                    assert (begin != nullptr);
                    assert (end != nullptr);
                    assert (begin <= end);
                    LOG_INFO("Pipe tail %p receives input iterator: size %d", this, (end - begin));
                    on_consume(begin, end);
                }
            };

        }
    }
}