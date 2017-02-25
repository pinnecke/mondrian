#pragma once

using namespace mondrian::query_engine::operators;

template<class InputType, class OutputType, class InputPointerType = InputType *,
        class OutputPointerType = OutputType *>
class generic_filter : public pipe<InputType, OutputType, InputPointerType, OutputPointerType> {
    using super = pipe<InputType, OutputType, InputPointerType, OutputPointerType>;
public:
    using typename super::input_t;
    using typename super::input_iterator_t;

private:
    function<bool(const input_iterator_t value)> predicate;

public:

    generic_filter(super *consumer, unsigned vector_size, function<bool(const input_iterator_t value)> predicate) :
            super(consumer, vector_size), predicate(predicate) { }

    virtual void on_consume(const input_iterator_t *begin, const input_iterator_t *end) override {
        for (auto it = begin; it != end; ++it) {
            if (predicate(super::as_reference(it)))
                super::forward(it);
        }
    }

};