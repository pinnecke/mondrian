#pragma once

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            template<class InputType, class InputPointerType = InputType*>
            class sink_operator : public push_operator<InputType, InputPointerType>
            {
                using super = push_operator<InputType, InputPointerType>;

            public:
                using typename super::input_t;
                using typename super::input_pointer_t;

                sink_operator() : super(nullptr, 0) {};

                virtual void on_produce() override final {}
            };

        }
    }
}

