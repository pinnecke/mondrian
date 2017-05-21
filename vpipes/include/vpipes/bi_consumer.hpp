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

#include <vpipes.hpp>


namespace mondrian{
    namespace vpipes{
            enum class which_operator { left_operator, right_operator, neither };
    }
}


namespace mondrian {
    namespace vpipes {
        template<class Input>
        class bi_consumer;
        template<class Input>
        class consumer_proxy : public consumer<Input>
        {

            using input_super = consumer<Input>;
            using bi_consumer_t = bi_consumer<Input>;

            public:
                using typename input_super::input_t;
                using typename input_super::input_batch_t;

            private:
                bi_consumer_t *context;
                which_operator operator_role;

            public:
                consumer_proxy(__in__ bi_consumer_t *context,
                               __in__ which_operator operator_role):
                        context(context), operator_role(operator_role)
                {

                }

                virtual const char *get_class_name() const override
                {
                    return "vpipes::pipes::consumer_proxy";
                }

            protected:
                inline virtual void on_consume(__in__ const input_batch_t *data) override __attribute__((always_inline))
                {
                    this->context->consume(data, this->operator_role);
                }

                inline virtual void on_cleanup() override __attribute__((always_inline))
                {
                    this->context->on_cleanup();
                }

        };
    }
}

namespace mondrian {
    namespace vpipes {
        template<class Input>
        class bi_consumer
        {

        using consumer_proxy_t = consumer_proxy<Input>;

        protected:
            statistics::operator_run statistics;
            consumer_proxy_t left_operator;
            consumer_proxy_t right_operator;

        public:
            using input_t = Input;
            using input_batch_t = batch<input_t>;

            friend
            class consumer_proxy<input_t>;

        protected:
            virtual void on_consume(__in__ const input_batch_t *data, __in__ which_operator operator_role) {};

            virtual void on_cleanup() {};

            inline virtual void consume(__in__ const input_batch_t *data, __in__ which_operator operator_role) final __attribute__((always_inline)) {
                    on_consume(data, operator_role);
            }

        public:
            bi_consumer() : left_operator(this, which_operator::left_operator),
                            right_operator(this, which_operator::right_operator)
            {

            }

            consumer_proxy_t *get_left_operator()
            {
                return &left_operator;
            }

            consumer_proxy_t *get_right_operator()
            {
                return &right_operator;
            }

            virtual void close()
            {
                on_cleanup();
            }

            const statistics::operator_run *get_input_statistics() const
            {
                return &statistics;
            }

            virtual const char *get_class_name() const = 0;

        };
    }
}