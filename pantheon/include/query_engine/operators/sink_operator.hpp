#pragma once

namespace mondrian {
    namespace query_engine {
        namespace operators {

            template<class ValueType>
            class sink_operator : public push_operator<ValueType> {
                using super = push_operator<ValueType>;
            public:
                sink_operator() : super(nullptr, 0) {};

                virtual void on_produce() override final {}
            };

        }
    }
}

