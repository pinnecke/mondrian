#pragma once

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            template<class InputType, class InputPointerType = InputType*>
            class source_operator : public push_operator<InputType, InputPointerType>
            {
                using super = push_operator<InputType, InputPointerType>;

            public:
                using typename super::input_t;
                using typename super::input_pointer_t;

            private:
                const input_pointer_t begin, end;

            protected:
                virtual const input_pointer_t get_begin() const final { return begin; }

                virtual const input_pointer_t get_end() const final { return end; }

            public:
                source_operator(push_operator <InputType, InputPointerType> *consumer, const input_pointer_t begin,
                                const input_pointer_t end, unsigned vector_size) :
                                super(consumer, vector_size), begin(begin), end(end) {};

                virtual void on_consume(const input_pointer_t *begin, const input_pointer_t *end) override final {};
            };

        }
    }
}

