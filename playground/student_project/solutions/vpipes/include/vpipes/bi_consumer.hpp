// Vector-Pipes - a framework for the push-based iterator model with support of vectorized execution
// Copyright (C) 2017  Marcus Pinnecke (marcus.pinnecke@ovgu.de)
//
// This program is free software; you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
// even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License al ong with this program; if
// not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
// Boston, MA  02110-1301, USA.

#pragma once

#include <functional>
#include "pipe_tail.hpp"
#include "vector.hpp"

using namespace std;

#define DEFINE_DELEGATE(FieldName, InputType, ForwardType, Method)                                                     \
template<class Input, class InputForwardIt>                                                                            \
class consume_delegate_##FieldName : public consumer<Input, InputForwardIt>                                            \
{                                                                                                                      \
    using typename consumer<Input, InputForwardIt>::input_iterator_t;                                                  \
    bi_consumer<InputLeft, InputRight, InputLeftForwardIt, InputRightForwardIt> *owner;                                \
public:                                                                                                                \
    consume_delegate_##FieldName(bi_consumer<InputLeft, InputRight, InputLeftForwardIt, InputRightForwardIt> *owner):  \
        owner(owner) { }                                                                                               \
    virtual void on_consume(const input_iterator_t *begin, const input_iterator_t *end) override                       \
    {   owner->Method(begin, end);  }                                                                                  \
};                                                                                                                     \
consume_delegate_##FieldName<InputType, ForwardType> FieldName;

namespace mondrian
{
    namespace vpipes
    {
        template<class InputLeft, class InputRight,
                    class InputLeftForwardIt = InputLeft*, class InputRightForwardIt = InputRight*>
        class bi_consumer
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
            virtual void on_consume_left(input_left_iterator_t *begin, input_left_iterator_t *end) = 0;
            virtual void on_consume_right(input_right_iterator_t *begin, input_right_iterator_t *end) = 0;

            virtual input_left_t lookup_left(input_left_iterator_t *ptr) final
            {
                return left_port.lookup(ptr);
            }

            virtual input_left_iterator_t as_reference_left(input_left_iterator_t *ptr) final
            {
                return left_port.as_reference(ptr);
            }

            virtual input_right_t lookup_right(input_right_iterator_t *ptr) final
            {
                return right_port.lookup(ptr);
            }

            virtual input_right_iterator_t as_reference_right(input_right_iterator_t *ptr) final
            {
                return right_port.as_reference(ptr);
            }

        public:
            bi_consumer(): left_port(this), right_port(this) { }

            bi_consumer<input_left_t, input_left_iterator_t> *get_left_port()
            {
                return &left_port;
            };

            bi_consumer<input_right_t, input_right_iterator_t> *get_right_port()
            {
                return &right_port;
            };

            virtual void close() { };
        };
    }
}