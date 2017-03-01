#pragma once

#include <functional>
#include "../pipe.hpp"

using namespace std;

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            namespace sql
            {

                /* Congratualtions, you find the solution for one task. Take this as a help for further tasks     *
                 *   - Marcus                                                                                     */

                template<class Type, class ForwardIt = Type*>
                class sequential_filter : public pipe<Type, Type, ForwardIt, ForwardIt>
                {
                    using super = pipe<Type, Type, ForwardIt, ForwardIt>;
                public:
                    using typename super::input_t;
                    using typename super::input_iterator_t;
                    using typename super::consumer_t;

                    std::function<bool(input_iterator_t)> predicate;
                public:

                    sequential_filter(consumer_t *consumer, unsigned vector_size,
                                      function<bool(input_iterator_t)> predicate) :
                            super(consumer, vector_size), predicate(predicate) { }

                    virtual void on_consume(input_iterator_t *begin, input_iterator_t *end) override
                    {
                        for (auto it = begin; it != end; ++it) {
                            if (predicate(super::as_reference(it)))
                                super::forward(it);
                        }
                    }
                };

            }
        }
    }
}

