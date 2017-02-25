#pragma once

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            template<class Input, class Output, class InputForwardIt = Input*, class OutputForwardIt = Output*>
            class pipe_tail : public pipe<Input, Output, InputForwardIt, OutputForwardIt>
            {
                using super = pipe<Input, Output, InputForwardIt, OutputForwardIt>;

            public:
                using typename super::input_t;
                using typename super::input_iterator_t;

                pipe_tail() : super(nullptr, 0) {};
            };

        }
    }
}

