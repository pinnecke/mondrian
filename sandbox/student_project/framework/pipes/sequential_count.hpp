#pragma once

#include <query_engine/operators/pipe.hpp>

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            namespace sql
            {

                /* Congratualtions, you find the solution for one task. Take this as a help for further tasks     *
                 *   - Marcus                                                                                     */

                template<class Input, class Output, class InputForwardIt = Input*, class OutputForwardIt = Output*>
                class sequential_count : public pipe<Input, Output, InputForwardIt, OutputForwardIt>
                {
                    using super = pipe<Input, Output, InputForwardIt, OutputForwardIt>;

                public:
                    using typename super::input_t;
                    using typename super::input_iterator_t;
                    using typename super::consumer_t;

                    input_iterator_t *list;
                    input_t count = 0;

                    sequential_count(consumer_t *consumer, unsigned vector_size) :
                            super(consumer, vector_size)
                    {
                        list = (input_iterator_t *) malloc(sizeof(input_iterator_t));
                    }

                    virtual void on_consume(input_iterator_t *begin, input_iterator_t *end) override
                    {
                        for (auto it = begin; it != end; ++it) {
                            count++;
                        }
                    }

                    virtual void on_close() override
                    {
                        list[0] = &count;
                        super::forward(list);
                    }

                    virtual void on_cleanup() override
                    {
                        free (list);
                    }
                };

            }
        }
    }
}

