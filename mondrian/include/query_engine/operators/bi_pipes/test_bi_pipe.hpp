#pragma once

#include <functional>
#include <query_engine/operators/bi_pipe.hpp>

using namespace std;

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            namespace sql
            {

                template<class InputLeft, class InputRight, class Output,
                        class InputLeftForwardIt = InputLeft*, class InputRightForwardIt = InputRight*,
                        class OutputForwardIt = Output*>
                class test_bi_pipe : public bi_pipe<InputLeft, InputRight, Output, InputLeftForwardIt,
                                                    InputRightForwardIt, OutputForwardIt>
                {
                    using super = bi_pipe<InputLeft, InputRight, Output, InputLeftForwardIt,
                            InputRightForwardIt, OutputForwardIt>;
                public:
                    using typename super::input_left_t;
                    using typename super::input_left_iterator_t;
                    using typename super::input_left_vector_t;
                    using typename super::input_right_t;
                    using typename super::input_right_iterator_t;
                    using typename super::input_right_vector_t;
                    using typename super::output_t;
                    using typename super::output_iterator_t;
                    using typename super::consumer_t;
                    using typename super::output_vector_t;

                public:

                    test_bi_pipe(consumer_t *consumer, unsigned vector_size) :
                                super(consumer, vector_size) { }

                    virtual void on_consume_left(const input_left_iterator_t *begin, const input_left_iterator_t *end) override {
                        // TODO: nested loop + full-consume (using .close() to call close(Left/Right) + delayed vector deallocation

                        std::cout << "consume left: " << (end - begin) << std::endl;
                    }

                    virtual void on_consume_right(const input_right_iterator_t *begin, const input_right_iterator_t *end) override {
                        std::cout << "consume right: " << (end - begin) << std::endl;
                    }



                };

            }
        }
    }
}

