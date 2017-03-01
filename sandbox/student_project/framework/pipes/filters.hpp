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
                template<class Type, class ForwardIt = Type*>
                class batched_pred_filter : public pipe<Type, Type, ForwardIt, ForwardIt>
                {
                    using super = pipe<Type, Type, ForwardIt, ForwardIt>;
                public:
                    using typename super::input_t;
                    using typename super::input_iterator_t;
                    using typename super::consumer_t;
                    using iterator_t = mondrian::query_engine::operators::iterator<input_iterator_t *>;
                    using predicate_function_t = function<void(input_iterator_t **result, size_t *result_size,
                                                               input_iterator_t *begin, input_iterator_t *end)>;

                private:
                    input_iterator_t *result_buffer;
                    size_t result_buffer_size;
                    predicate_function_t predicate;
                public:

                    batched_pred_filter(consumer_t *consumer, unsigned vector_size, predicate_function_t predicate) :
                            super(consumer, vector_size), predicate(predicate)
                    {
                        // Note here: The operator is unaware of the vector size of the input. The assignment
                        // of the vector size of this operator as the vector size of the preceding operator
                        // just a best guess and must be corrected afterwards if it was wrong
                        result_buffer_size = vector_size;
                        result_buffer = (input_iterator_t *) malloc(result_buffer_size * sizeof(input_iterator_t));
                        assert (result_buffer != nullptr);
                    }

                    virtual void on_consume(input_iterator_t *begin, input_iterator_t *end) override
                    {
                        auto prec_vec_size = (end - begin);
                        if (prec_vec_size > result_buffer_size) {
                            result_buffer_size = prec_vec_size;
                            result_buffer = (input_iterator_t *) realloc(result_buffer, prec_vec_size *
                                    sizeof(input_iterator_t));
                            assert (result_buffer != nullptr);
                        }
                        size_t result_size;
                        predicate(&result_buffer, &result_size, begin, end);
                        super::produce(result_buffer, result_buffer + result_size);
                    }

                    virtual void on_cleanup() override
                    {
                        free (result_buffer);
                    }
                };

                template<class Type, class ForwardIt = Type*>
                class simple_filter : public pipe<Type, Type, ForwardIt, ForwardIt>
                {
                    using super = pipe<Type, Type, ForwardIt, ForwardIt>;
                public:
                    using typename super::input_t;
                    using typename super::input_iterator_t;
                    using typename super::consumer_t;

                    std::function<bool(input_iterator_t)> predicate;
                public:

                    simple_filter(consumer_t *consumer, unsigned vector_size,
                                      function<bool(input_iterator_t)> predicate) :
                            super(consumer, vector_size), predicate(predicate) { }

                    virtual void on_consume(input_iterator_t *begin, input_iterator_t *end) override
                    {
                        for (auto it = begin; it != end; ++it) {
                            if (predicate(super::as_reference(it)))
                                super::produce(it);
                        }
                    }
                };

            }
        }
    }
}

