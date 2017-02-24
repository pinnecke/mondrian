#pragma once

using namespace mondrian::query_engine::operators;

template<class InputType, class OutputType, class InputPointerType = InputType *,
        class OutputPointerType = OutputType *>
class counter : public push_operator<InputType, OutputType, InputPointerType, OutputPointerType> {
    using super = push_operator<InputType, OutputType, InputPointerType, OutputPointerType>;
public:
    using typename super::input_t;
    using typename super::input_iterator_t;

private:
    input_iterator_t *list;
    input_t count = 0;

public:
    counter(super *consumer, unsigned vector_size) :
            super(consumer, vector_size) {
        list = (input_iterator_t *) malloc(sizeof(input_iterator_t));
    }

    virtual void on_consume(const input_iterator_t *begin, const input_iterator_t *end) override {
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