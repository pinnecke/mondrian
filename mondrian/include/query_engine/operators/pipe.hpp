#pragma once

#include <query_engine/operators/pipe_tail.hpp>

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            template<class Input, class Output, class InputForwardIt = Input*, class OutputForwardIt = Output*>
            class pipe : public pipe_tail<Input, InputForwardIt>
            {
                using super = pipe_tail<Input, InputForwardIt>;
            public:
                using typename super::input_t;
                using typename super::input_iterator_t;
                using typename super::input_vector_t;

                using output_t = Output;
                using output_iterator_t = OutputForwardIt;
                using consumer_t = pipe_tail<output_t, output_iterator_t>;

                using output_vector_t = vector<output_t, output_iterator_t>;

            private:
                consumer_t *consumer;
                output_vector_t *result = nullptr;
                size_t size;

                void reset() { result = new output_vector_t(size); }

                void cleanup()
                {
                    if (result != nullptr) {
                        result->release();
                        delete (result);
                        result = nullptr;
                    }
                }

                void send()
                {
                    consumer->consume(result);
                    cleanup();
                }

            protected:

                virtual void on_close() { };

                virtual void on_cleanup() { };

                virtual void forward(const output_iterator_t *value) final
                {
                    if (result->add(*value) == output_vector_t::state::full) {
                        send();
                        reset();
                    }
                }

                virtual void close() override final
                {
                    if (consumer != nullptr)
                    {
                        send();
                        reset();
                        on_close();
                        send();
                        consumer->close();
                        on_cleanup();
                    }
                    cleanup();
                }

                virtual input_t lookup(const input_iterator_t *ptr) final
                {
                    return **ptr;
                }

                virtual const input_iterator_t as_reference(const input_iterator_t *ptr) final
                {
                    return *ptr;
                }

            public:
                pipe(consumer_t *consumer, unsigned vector_size):
                        consumer(consumer), size(vector_size)
                {
                    reset();
                }
            };

        }
    }
}
