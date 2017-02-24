#pragma once

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            template<class Input, class Output, class InputForwardIt = Input*, class OutputForwardIt = Output*>
            class sink_operator : public push_operator<Input, Output, InputForwardIt, OutputForwardIt>
            {
                using super = push_operator<Input, Output, InputForwardIt, OutputForwardIt>;

            public:
                using typename super::input_t;
                using typename super::input_iterator_t;

                sink_operator() : super(nullptr, 0) {};
            };

        }
    }
}

