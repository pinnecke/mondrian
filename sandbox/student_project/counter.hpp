#pragma once

using namespace mondrian::query_engine::operators;

template<class InputType, class InputPointerType = InputType*>
class counter : public push_operator<InputType, InputPointerType> {
    using super = push_operator<InputType, InputPointerType>;
public:
    using typename super::input_t;
    using typename super::input_pointer_t;

private:
    input_pointer_t *list;
    input_t count = 0;

public:
    counter(super *consumer, unsigned vector_size) :
            super(consumer, vector_size) {
        list = (input_pointer_t*) malloc(sizeof(input_pointer_t));
    }

    virtual void on_consume(const input_pointer_t*begin, const input_pointer_t *end) override {
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