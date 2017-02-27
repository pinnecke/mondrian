#pragma once

#include <query_engine/operators/forwarder.hpp>
#include <query_engine/operators/pipe_tail.hpp>

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            template<class Input, class Output, class InputForwardIt = Input*, class OutputForwardIt = Output*>
            class pipe : public pipe_tail<Input, InputForwardIt>, forwarder<Output, OutputForwardIt>
            {
                using input_super = pipe_tail<Input, InputForwardIt>;
                using output_super = forwarder<Output, OutputForwardIt>;

            public:
                using typename input_super::input_t;
                using typename input_super::input_iterator_t;
                using typename input_super::input_vector_t;

                using typename output_super::output_t;
                using typename output_super::output_iterator_t;
                using typename output_super::output_vector_t;
                using typename output_super::consumer_t;

            protected:
                using output_super::forward;
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
