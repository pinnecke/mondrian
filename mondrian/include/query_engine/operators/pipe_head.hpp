#pragma once

#include <query_engine/operators/pipe.hpp>

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            template<class Output, class OutputForwardIt = Output*>
            class pipe_head : public pipe<Output, Output, OutputForwardIt, OutputForwardIt>
            {
                using super = pipe<Output, Output, OutputForwardIt, OutputForwardIt>;

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
                pipe_head(consumer_t *consumer, const input_iterator_t begin, const input_iterator_t end,
                                unsigned vector_size) :
                                super(consumer, vector_size), begin(begin), end(end) {};

                virtual void on_consume(const input_iterator_t *begin, const input_iterator_t *end) override final {};

                virtual void produce() final { on_produce(); }
            };

        }
    }
}

