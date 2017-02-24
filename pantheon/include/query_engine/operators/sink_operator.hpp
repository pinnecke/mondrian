#pragma once

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            template<class InputType, class OutputType, class InputPointerType = InputType *,
                    class OutputPointerType = OutputType *>
            class sink_operator : public push_operator<InputType, OutputType, InputPointerType, OutputPointerType>
            {
                using super = push_operator<InputType, OutputType, InputPointerType, OutputPointerType>;

            public:
                using typename super::input_t;
                using typename super::input_pointer_t;

                sink_operator() : super(nullptr, 0) {};
            };

        }
    }
}

