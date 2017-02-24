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

                template<class InputType, class InputPointerType = InputType*>
                class printer : public sink_operator<InputType, InputPointerType>
                {
                    using super = sink_operator<InputType, InputPointerType>;

                public:
                    using typename super::input_t;
                    using typename super::input_pointer_t;

                    printer() : super() {};

                    virtual void on_consume(const input_pointer_t *begin, const input_pointer_t *end) override
                    {
                        for (auto it = begin; it != end; ++it)
                            std::cout << ">> " << **it << std::endl;
                    }
                };

            }
        }
    }
}

