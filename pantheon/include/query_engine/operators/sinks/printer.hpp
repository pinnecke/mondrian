#pragma once

#include <query_engine/operators/sink_operator.hpp>

namespace mondrian {
    namespace query_engine {
        namespace operators {
            namespace sinks {

                template<class ValueType>
                class printer : public sink_operator<ValueType> {
                    using super = sink_operator<ValueType>;
                public:
                    printer() : super() {};

                    virtual void on_consume(const ValueType **begin, const ValueType **end) override {
                        for (auto it = begin; it != end; ++it)
                            std::cout << ">> " << **it << std::endl;
                    }
                };

            }
        }
    }
}

