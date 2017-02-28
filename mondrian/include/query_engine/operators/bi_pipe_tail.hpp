#pragma once

#include <functional>
#include <query_engine/operators/pipe_tail.hpp>
#include <query_engine/operators/vector.hpp>

using namespace std;

#define DEFINE_DELEGATE(FieldName, InputType, ForwardType, Method)                                                     \
template<class Input, class InputForwardIt>                                                                            \
class consume_delegate_##FieldName : public pipe_tail<Input, InputForwardIt>                                           \
{                                                                                                                      \
    using typename pipe_tail<Input, InputForwardIt>::input_iterator_t;                                                 \
    bi_pipe_tail<InputLeft, InputRight, InputLeftForwardIt, InputRightForwardIt> *owner;                               \
public:                                                                                                                \
    consume_delegate_##FieldName(bi_pipe_tail<InputLeft, InputRight, InputLeftForwardIt, InputRightForwardIt> *owner): \
        owner(owner) { }                                                                                               \
    virtual void on_consume(const input_iterator_t *begin, const input_iterator_t *end) override                       \
    {   owner->Method(begin, end);  }                                                                                  \
};                                                                                                                     \
consume_delegate_##FieldName<InputType, ForwardType> FieldName;

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            template<class InputLeft, class InputRight,
                    class InputLeftForwardIt = InputLeft*, class InputRightForwardIt = InputRight*>
            class bi_pipe_tail
            {
                DEFINE_DELEGATE(left_port, InputLeft, InputLeftForwardIt, on_consume_left);
                DEFINE_DELEGATE(right_port, InputRight, InputRightForwardIt, on_consume_right);

            public:
                using input_left_t = InputLeft;
                using input_left_iterator_t = InputLeftForwardIt;
                using input_left_vector_t = vector<input_left_t, input_left_iterator_t>;

                using input_right_t = InputRight;
                using input_right_iterator_t = InputRightForwardIt;
                using input_right_vector_t = vector<input_right_t, input_right_iterator_t>;

            protected:
                virtual void on_consume_left(const input_left_iterator_t *begin, const input_left_iterator_t *end) = 0;
                virtual void on_consume_right(const input_right_iterator_t *begin, const input_right_iterator_t *end) = 0;

                virtual input_left_t lookup_left(const input_left_iterator_t *ptr) final
                {
                    return left_port.lookup(ptr);
                }

                virtual const input_left_iterator_t as_reference_left(const input_left_iterator_t *ptr) final
                {
                    return left_port.as_reference(ptr);
                }

                virtual input_right_t lookup_right(const input_right_iterator_t *ptr) final
                {
                    return right_port.lookup(ptr);
                }

                virtual const input_right_iterator_t as_reference_right(const input_right_iterator_t *ptr) final
                {
                    return right_port.as_reference(ptr);
                }

            public:
                bi_pipe_tail(): left_port(this), right_port(this) { }

                pipe_tail<input_left_t, input_left_iterator_t> *get_left_port()
                {
                    return &left_port;
                };

                pipe_tail<input_right_t, input_right_iterator_t> *get_right_port()
                {
                    return &right_port;
                };

                virtual void close() { };
            };

        }
    }
}