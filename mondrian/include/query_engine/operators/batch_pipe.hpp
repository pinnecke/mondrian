#pragma once

#include <query_engine/operators/pipe_tail.hpp>

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            template<class Input, class Output, class InputForwardIt = Input*, class OutputForwardIt = Output*>
            class batch_pipe : public pipe_tail<Input, InputForwardIt>
            {
                using super = pipe_tail<Input, InputForwardIt>;
            public:
                using typename super::input_t;
                using typename super::input_iterator_t;
                using typename super::input_vector_t;

                using output_t = Output;
                using output_iterator_t = OutputForwardIt;
                using consumer_t = pipe_tail<output_t, output_iterator_t>;

            private:
                consumer_t *consumer;
                input_iterator_t *batch = nullptr;
                size_t size, capacity;

                void cleanup()
                {
                    free (batch);
                    batch = nullptr;
                }

            protected:
                virtual void on_consume(const input_iterator_t *begin, const input_iterator_t *end) final override
                {
                    auto dist = distance(begin, end);
                    auto next_size = dist + size + 1;
                    if (next_size >= capacity) {
                        while (next_size >= capacity)
                            capacity *= 1.4f;
                        batch = (input_iterator_t *) realloc(batch, capacity * sizeof(input_iterator_t));
                        assert (batch != nullptr);
                    }
                    for (auto it = begin; it < end; ++it) {
                        batch[size++] = *it;
                    }
                }

                virtual void on_batch_process(const output_iterator_t **output_begin, const output_iterator_t **output_end,
                                              const input_iterator_t *begin, const input_iterator_t *end) = 0;

                virtual void on_cleanup() { };

                virtual void close() override final
                {
                    if (consumer != nullptr)
                    {
                        auto input_begin = batch;
                        auto input_end = input_begin + size;
                        const output_iterator_t *output_begin, *output_end;

                        on_batch_process(&output_begin, &output_end, input_begin, input_end);
                        assert (output_begin != nullptr);
                        assert (output_end != nullptr);
                        assert (output_begin <= output_end);

                        consumer->consume(output_begin, output_end);
                        consumer->close();
                    }
                    on_cleanup();
                    cleanup();
                }

            public:
                batch_pipe(consumer_t *consumer, unsigned initial_capacity):
                        consumer(consumer), capacity(initial_capacity), size(0)
                {
                    batch = (input_iterator_t *) malloc (initial_capacity * sizeof(input_iterator_t));
                    assert (batch != nullptr);
                }
            };

        }
    }
}
