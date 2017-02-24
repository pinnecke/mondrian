#pragma once

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            template<class InputType, class OutputType, class InputPointerType = InputType *,
                    class OutputPointerType = OutputType *>
            class source_operator : public push_operator<InputType, OutputType, InputPointerType, OutputPointerType>
            {
                using super = push_operator<InputType, OutputType, InputPointerType, OutputPointerType>;

            public:
                using typename super::input_t;
                using typename super::input_pointer_t;
                using typename super::consumer_t;

            private:
                const input_pointer_t begin, end;

            protected:
                virtual void on_produce() = 0;

                virtual const input_pointer_t get_begin() const final { return begin; }

                virtual const input_pointer_t get_end() const final { return end; }

            public:
                source_operator(consumer_t *consumer, const input_pointer_t begin,
                                const input_pointer_t end, unsigned vector_size) :
                                super(consumer, vector_size), begin(begin), end(end) {};

                virtual void on_consume(const input_pointer_t *begin, const input_pointer_t *end) override final {};

                virtual void produce() final { on_produce(); }
            };

        }
    }
}

