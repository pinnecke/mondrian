#pragma once

#include "../pipe_tail.hpp"

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            namespace sinks
            {
                template<class Input, class InputForwardIt = Input *>
                class materialize : public consumer<Input, InputForwardIt>
                {
                    using super = consumer<Input, InputForwardIt>;
                    size_t i;
                    InputForwardIt destination;

                public:
                    using typename super::input_t;
                    using typename super::input_iterator_t;

                    materialize(InputForwardIt destination) : super(), destination(destination), i(0) {};

                    virtual void on_consume(input_iterator_t *begin, input_iterator_t *end) override
                    {
                        for (auto it = begin; it != end; ++it)
                            destination[i] = super::lookup(it);
                    }
                };

            }
        }
    }
}

