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
                template<class Input, class InputForwardIt = Input*>
                class printer : public consumer<Input, InputForwardIt>
                {
                    using super = consumer<Input, InputForwardIt>;

                public:
                    using typename super::input_t;
                    using typename super::input_iterator_t;

                    printer() : super() {};

                    virtual void on_consume(input_iterator_t *begin, input_iterator_t *end) override
                    {
                        for (auto it = begin; it != end; ++it)
                            std::cout << ">> " << **it << std::endl;
                    }
                };

            }
        }
    }
}

