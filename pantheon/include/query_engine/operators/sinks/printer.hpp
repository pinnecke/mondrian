#pragma once

#include <query_engine/operators/sink_operator.hpp>

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            namespace sinks
            {
                template<class Input, class Output, class InputForwardIt = Input*, class OutputForwardIt = Output*>
                class printer : public sink_operator<Input, Output, InputForwardIt, OutputForwardIt>
                {
                    using super = sink_operator<Input, Output, InputForwardIt, OutputForwardIt>;

                public:
                    using typename super::input_t;
                    using typename super::input_iterator_t;

                    printer() : super() {};

                    virtual void on_consume(const input_iterator_t *begin, const input_iterator_t *end) override
                    {
                        for (auto it = begin; it != end; ++it)
                            std::cout << ">> " << **it << std::endl;
                    }
                };

            }
        }
    }
}

