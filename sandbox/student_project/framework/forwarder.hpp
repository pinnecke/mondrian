#pragma once

#include "pipe_tail.hpp"

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            template<class Output, class OutputForwardIt = Output*>
            class forwarder
            {
            public:
                using output_t = Output;
                using output_iterator_t = OutputForwardIt;
                using consumer_t = pipe_tail<output_t, output_iterator_t>;
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

                virtual void forward(output_iterator_t *value) final
                {
                    if (result->add(*value) == output_vector_t::state::full) {
                        send();
                        reset();
                    }
                }

            public:
                forwarder(consumer_t *consumer, unsigned vector_size):
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
