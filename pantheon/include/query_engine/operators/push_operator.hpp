#pragma once

#include <query_engine/operators/consumer.hpp>

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            template<class Input, class Output, class InputForwardIt = Input*, class OutputForwardIt = Output*>
            class push_operator : public consumer<Input, InputForwardIt>
            {
                using super = consumer<Input, InputForwardIt>;
            public:
                using typename super::input_t;
                using typename super::input_iterator_t;
                using typename super::input_vector_t;

                using output_t = Output;
                using output_iterator_t = OutputForwardIt;
                using consumer_t = consumer<output_t, output_iterator_t>;

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

                virtual void on_consume(const input_iterator_t *begin, const input_iterator_t *end) { };

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
                push_operator(consumer_t *consumer, unsigned vector_size) :
                        consumer(consumer), size(vector_size)
                {
                    reset();
                }

                virtual void consume(input_vector_t *data) override final
                {
                    auto iterator = data->get_iterator();
                    if (!iterator.is_empty())
                        on_consume(iterator.begin, iterator.end);
                }
            };

        }
    }
}