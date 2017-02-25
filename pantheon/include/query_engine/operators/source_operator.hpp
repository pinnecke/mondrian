#pragma once

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            template<class Input, class Output, class InputForwardIt = Input*, class OutputForwardIt = Output*>
            class source_operator : public pipe<Input, Output, InputForwardIt, OutputForwardIt>
            {
                using super = pipe<Input, Output, InputForwardIt, OutputForwardIt>;

            public:
                using typename super::input_t;
                using typename super::input_iterator_t;
                using typename super::consumer_t;

            private:
                const input_iterator_t begin, end;

            protected:
                virtual void on_produce() = 0;

                virtual const input_iterator_t get_begin() const final { return begin; }

                virtual const input_iterator_t get_end() const final { return end; }

            public:
                source_operator(consumer_t *consumer, const input_iterator_t begin, const input_iterator_t end,
                                unsigned vector_size) :
                                super(consumer, vector_size), begin(begin), end(end) {};

                virtual void on_consume(const input_iterator_t *begin, const input_iterator_t *end) override final {};

                virtual void produce() final { on_produce(); }
            };

        }
    }
}

