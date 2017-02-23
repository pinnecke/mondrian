#pragma once

using namespace mondrian::query_engine::operators;

template<class ValueType>
class counter : public push_operator<ValueType> {
    using super = push_operator<ValueType>;
public:
    const ValueType **list;
    ValueType count = 0;

    counter(super *consumer, unsigned vector_size) :
            super(consumer, vector_size) {
        list = (const ValueType **) malloc(sizeof(ValueType*));
    }

    virtual void on_consume(const ValueType **begin, const ValueType **end) override {
        for (auto it = begin; it != end; ++it) {
            count++;
        }
    }

    virtual void on_close() override {
        list[0] = &count;
        super::forward(list);
    }

    virtual void on_cleanup() override {
        free (list);
    }
};