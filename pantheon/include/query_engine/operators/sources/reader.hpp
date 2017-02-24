#pragma once

#include <query_engine/operators/source_operator.hpp>

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            namespace sources
            {
                template<class Input, class Output, class InputForwardIt = Input*, class OutputForwardsIt = Output*>
                class reader : public source_operator<Input, Output, InputForwardIt, OutputForwardsIt>
                {
                    using super = source_operator<Input, Output, InputForwardIt, OutputForwardsIt>;

                public:
                    using typename super::input_t;
                    using typename super::input_iterator_t;
                    using typename super::consumer_t;

                    reader(consumer_t *consumer, const input_iterator_t begin, const input_iterator_t end,
                           unsigned vector_size): super(consumer, begin, end, vector_size) {}

                    virtual void on_produce() override
                    {
                        auto begin = super::get_begin();
                        auto end = super::get_end();
                        for (auto it = begin; it != end; ++it)
                            super::forward(&it);
                        super::close();
                    };
                };

            }
        }
    }
}

