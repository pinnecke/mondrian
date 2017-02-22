#pragma once

#include <query_engine/operators/push_operator.hpp>

namespace mondrian {
    namespace query_engine {
        namespace operators {
            namespace sql {

                template<class ValueType>
                class sequential_scan : public push_operator<ValueType> {
                    using super = push_operator<ValueType>;
                public:
                    sequential_scan(super *consumer, unsigned vector_size) :
                            super(consumer, vector_size) {}

                    virtual void on_consume(const ValueType **begin, const ValueType **end) override {
                        for (auto it = begin; it != end; ++it) {
                            if (super::lookup(it) < 5)
                                super::yield(it);
                        }
                    }
                };

            }
        }
    }
}

