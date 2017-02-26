#pragma once

#include <query_engine/operators/pipe_head.hpp>

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            namespace sources
            {
                template<class Output, class OutputForwardsIt = Output*>
                class reader : public pipe_head<Output, OutputForwardsIt>
                {
                    using super = pipe_head<Output, OutputForwardsIt>;

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

