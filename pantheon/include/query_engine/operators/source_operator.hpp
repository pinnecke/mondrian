#pragma once

namespace mondrian {
    namespace query_engine {
        namespace operators {

            template<class ValueType>
            class source_operator : public push_operator<ValueType> {
                using super = push_operator<ValueType>;
            private:
                const ValueType *begin, *end;
            protected:
                virtual const ValueType *get_begin() const final { return begin; }

                virtual const ValueType *get_end() const final { return end; }

            public:
                source_operator(push_operator <ValueType> *consumer, const ValueType *begin, const ValueType *end,
                                unsigned vector_size) : super(consumer, vector_size), begin(begin), end(end) {};

                virtual void on_consume(const ValueType **begin, const ValueType **end) override final {};
            };

        }
    }
}

