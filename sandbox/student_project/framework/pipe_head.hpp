#pragma once

#include "pipe.hpp"

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
                input_iterator_t begin, end;

            protected:
                virtual void on_produce() = 0;

                virtual input_iterator_t get_begin() final { return begin; }

                virtual input_iterator_t get_end() final { return end; }

            public:
                pipe_head(consumer_t *consumer, input_iterator_t begin, input_iterator_t end,
                                unsigned vector_size) :
                                super(consumer, vector_size), begin(begin), end(end) {};

                virtual void on_consume(input_iterator_t *begin, input_iterator_t *end) override final {};

                virtual void produce() final { on_produce(); }
            };

        }
    }
}

