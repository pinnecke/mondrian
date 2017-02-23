#pragma once

using namespace mondrian::query_engine::operators;

template<class ValueType>
class generic_filter : public push_operator<ValueType> {
    using super = push_operator<ValueType>;
    function<bool(const ValueType *value)> predicate;
public:

    generic_filter(super *consumer, unsigned vector_size, function<bool(const ValueType *value)> predicate) :
            super(consumer, vector_size), predicate(predicate) { }

    virtual void on_consume(const ValueType **begin, const ValueType **end) override {
        for (auto it = begin; it != end; ++it) {
            if (predicate(super::as_reference(it)))
                super::forward(it);
        }
    }

};