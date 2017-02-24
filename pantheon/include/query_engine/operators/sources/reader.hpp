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
                template<class InputType, class OutputType, class InputPointerType = InputType *,
                        class OutputPointerType = OutputType *>
                class reader : public source_operator<InputType, OutputType, InputPointerType,
                        OutputPointerType>
                {
                    using super = source_operator<InputType, OutputType, InputPointerType, OutputPointerType>;

                public:
                    using typename super::input_t;
                    using typename super::input_pointer_t;
                    using typename super::consumer_t;

                    reader(consumer_t *consumer, const input_pointer_t begin,
                           const input_pointer_t end, unsigned vector_size) :
                            super(consumer, begin, end, vector_size) {}

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

