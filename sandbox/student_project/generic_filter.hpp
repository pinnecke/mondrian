#pragma once

using namespace mondrian::query_engine::operators;

template<class InputType, class InputPointerType = InputType*>
class generic_filter : public push_operator<InputType, InputPointerType> {
    using super = push_operator<InputType, InputPointerType>;
public:
    using typename super::input_t;
    using typename super::input_pointer_t;

private:
    function<bool(const input_pointer_t value)> predicate;

public:

    generic_filter(super *consumer, unsigned vector_size, function<bool(const input_pointer_t value)> predicate) :
            super(consumer, vector_size), predicate(predicate) { }

    virtual void on_consume(const input_pointer_t *begin, const input_pointer_t *end) override {
        for (auto it = begin; it != end; ++it) {
            if (predicate(super::as_reference(it)))
                super::forward(it);
        }
    }

};