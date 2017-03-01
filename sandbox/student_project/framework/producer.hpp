#pragma once

#include "pipe_tail.hpp"

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            template<class Output, class OutputForwardIt = Output*>
            class producer
            {
            public:
                using output_t = Output;
                using output_iterator_t = OutputForwardIt;
                using consumer_t = consumer<output_t, output_iterator_t>;
                using output_vector_t = vector<output_t, output_iterator_t>;

            private:
                consumer_t *consumer;
                output_vector_t *result = nullptr;
                size_t size;

            protected:

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

                virtual void produce(output_iterator_t *value) final
                {
                    produce(value, value + 1);
                }

                virtual inline void produce(output_iterator_t *begin, output_iterator_t *end) final
                {
                    do {
                        typename output_vector_t::state vector_state;
                        begin = result->add(&vector_state, begin, end);
                        if (vector_state == output_vector_t::state::full) {
                            send();
                            reset();
                        }
                    } while (begin != end);
                }

            public:
                producer(consumer_t *consumer, unsigned vector_size):
                        consumer(consumer), size(vector_size)
                {
                    reset();
                }

                virtual void close()
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
            };

        }
    }
}
