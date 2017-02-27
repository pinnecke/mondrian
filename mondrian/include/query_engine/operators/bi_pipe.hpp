#pragma once

#include <query_engine/operators/forwarder.hpp>
#include <query_engine/operators/bi_pipe_tail.hpp>
#include <query_engine/operators/pipe_tail.hpp>

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            template<class InputLeft, class InputRight, class Output,
                     class InputLeftForwardIt = InputLeft*, class InputRightForwardIt = InputRight*,
                     class OutputForwardIt = Output*>
            class bi_pipe : public bi_pipe_tail<InputLeft, InputRight, InputLeftForwardIt, InputRightForwardIt>,
                                   forwarder<Output, OutputForwardIt>
            {
                using input_super = bi_pipe_tail<InputLeft, InputRight, InputLeftForwardIt, InputRightForwardIt>;
                using output_super = forwarder<Output, OutputForwardIt>;
            public:
                using typename input_super::input_left_t;
                using typename input_super::input_left_iterator_t;
                using typename input_super::input_left_vector_t;
                using typename input_super::input_right_t;
                using typename input_super::input_right_iterator_t;
                using typename input_super::input_right_vector_t;

                using typename output_super::output_t;
                using typename output_super::output_iterator_t;
                using typename output_super::consumer_t;
                using typename output_super::output_vector_t;

                using output_super::forward;
                using input_super::lookup_left;
                using input_super::lookup_right;
                using input_super::as_reference_left;
                using input_super::as_reference_right;

            public:
                bi_pipe(consumer_t *consumer, unsigned vector_size):
                        output_super(consumer, vector_size) { }
            };

        }
    }
}
