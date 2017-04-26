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

using namespace std;

namespace mondrian
{
    namespace vpipes
    {
        namespace pipes
        {
            template<class Input>
            class filter : public pipe<Input, Input>
            {
                using super = pipe<Input, Input>;
            public:
                using typename super::input_t;
                using typename super::input_batch_t;
                using typename super::output_t;
                using typename super::output_batch_t;
                using typename super::consumer_t;

                using iterator_t = vpipes::iterator<tuplet_id_t *>;
                using predicate_func_t = typename vpipes::predicates::batched_predicates<input_t>::func_t;

            private:
                size_t *matching_indices_buffer;
                size_t buffer_size;
                bool hint_avg_batch_eval_result_is_non_empty;
                statistics::predicate_run statistics;
                null_value_filter_policy null_policy;

                predicate_func_t predicate;
            public:

                filter(__in__ consumer_t *destination,
                       __in__ predicate_func_t predicate,
                       __in__ null_value_filter_policy null_policy,
                       __in__ unsigned batch_size,
                       __in__ bool hint_avg_batch_eval_result_is_non_empty) :
                        super(destination, batch_size), predicate(predicate), null_policy(null_policy),
                        hint_avg_batch_eval_result_is_non_empty(hint_avg_batch_eval_result_is_non_empty)
                {
                    // Note here: The operator is unaware of the batch size of the input. The assignment
                    // of the batch size of this operator as the batch size of the preceding operator
                    // just a best guess and must be corrected afterwards if it was wrong
                    buffer_size = batch_size;
                    matching_indices_buffer = (size_t *) malloc(buffer_size * sizeof(size_t));
                    assert (matching_indices_buffer != nullptr);
                }

                inline virtual void on_consume(__in__ const input_batch_t *data) override final __attribute__((always_inline))
                {
                    auto input_batch_size = data->get_size();

                    if (__builtin_expect(input_batch_size > buffer_size, false)) {
                        buffer_size = input_batch_size;
                        matching_indices_buffer = (size_t *) realloc(matching_indices_buffer, input_batch_size *
                                                                     sizeof(tuplet_id_t));
                        assert (matching_indices_buffer != nullptr);
                    }

                    size_t result_size = 0;
                    predicate(matching_indices_buffer, &result_size, &statistics, null_policy,
                              data->get_tupletids(), data->get_values(), data->get_null_mask(), input_batch_size);
                    assert (result_size <= buffer_size);

                    if (__builtin_expect(result_size != 0, hint_avg_batch_eval_result_is_non_empty)) {
                        super::produce(data->get_tupletids(), data->get_values(), data->get_null_mask(),
                                       matching_indices_buffer, result_size, false);
                    }
                }

                virtual void on_cleanup() override
                {
                    free (matching_indices_buffer);
                }

                const statistics::predicate_run *get_predicate_statistics() const
                {
                    return &this->statistics;
                }

                virtual const char *get_class_name() const override
                {
                    return "vpipes::pipes::filter";
                }
            };
        }
    }
}

