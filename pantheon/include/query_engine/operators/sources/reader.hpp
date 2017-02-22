#pragma once

#include <query_engine/operators/source_operator.hpp>

namespace mondrian {
    namespace query_engine {
        namespace operators {
            namespace sources {

                template<class ValueType>
                class reader : public source_operator<ValueType> {
                    using super = source_operator<ValueType>;
                public:
                    reader(push_operator <ValueType> *consumer, const ValueType *begin, const ValueType *end,
                           unsigned vector_size) :
                            super(consumer, begin, end, vector_size) {}

                    virtual void on_produce() override {
                        auto begin = super::get_begin();
                        auto end = super::get_end();
                        for (auto it = begin; it != end; ++it)
                            super::yield(&it);
                        super::close();
                    };
                };

            }
        }
    }
}

