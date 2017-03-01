#pragma once

#include "producer.hpp"
#include "pipe_tail.hpp"

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            template<class Input, class Output, class InputForwardIt = Input*, class OutputForwardIt = Output*>
            class pipe : public consumer<Input, InputForwardIt>, public producer<Output, OutputForwardIt>
            {
                using input_super = consumer<Input, InputForwardIt>;
                using output_super = producer<Output, OutputForwardIt>;

            public:
                using typename input_super::input_t;
                using typename input_super::input_iterator_t;
                using typename input_super::input_vector_t;

                using typename output_super::output_t;
                using typename output_super::output_iterator_t;
                using typename output_super::output_vector_t;
                using typename output_super::consumer_t;

            protected:
                using output_super::produce;
                using input_super::lookup;
                using input_super::as_reference;

            public:
                pipe(consumer_t *consumer, unsigned vector_size):
                        output_super(consumer, vector_size) { }

                virtual void close() final
                {
                    output_super::close();
                }
            };

        }
    }
}
