#pragma once

#include <query_engine/operators/batch_pipe.hpp>

using namespace std;

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            namespace sql
            {
                template<class Input, class InputForwardIt = Input*>
                class collector : public batch_pipe<Input, Input, InputForwardIt, InputForwardIt>
                {
                    using super = batch_pipe<Input, Input, InputForwardIt, InputForwardIt>;
                public:
                    using typename super::input_t;
                    using typename super::input_iterator_t;
                    using typename super::consumer_t;

                protected:
                    virtual void on_batch_process(input_iterator_t **output_begin, input_iterator_t **output_end,
                                                  input_iterator_t *begin, input_iterator_t *end) override
                    {
                        assert (begin != nullptr);
                        assert (end != nullptr);
                        assert (begin <= end);

                        *output_begin = begin;
                        *output_end = end;
                    }

                public:
                    collector(consumer_t *consumer, unsigned initial_capacity = 10000) :
                        super(consumer, initial_capacity) { }
                };

            }
        }
    }
}