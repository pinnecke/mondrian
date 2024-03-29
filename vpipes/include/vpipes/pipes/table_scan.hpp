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
            template <class InputType, class InputTupletIdType = size_t>
            class table_scan : public producer<InputType>
            {
                using super = producer<InputType>;

            public:
                using input_t = InputType;
                using typename super::consumer_t;
                using filter_t = filter<input_t>;
                using predicate_func_t = typename filter_t::predicate_func_t;
                using block_copy_t = typename block_copy<input_t>::func_t;
                using block_null_copy_t = typename block_null_copy::func_t;
                using interval_t = interval<InputTupletIdType>;

            private:
                const interval_t *tuplet_ids_interval_begin, *tuplet_ids_interval_end;
                filter_t *filter_operator;
                block_copy_t block_copy_func;
                block_null_copy_t block_null_copy_func;

            public:
                table_scan(__in__ consumer_t *destination,
                           __in__ const interval_t *tuplet_ids_interval_begin,
                           __in__ const interval_t *tuplet_ids_interval_end,
                           __in__ predicate_func_t predicate,
                           __in__ null_value_filter_policy null_policy,
                           __in__ block_copy_t block_copy_func,
                           __in__ block_null_copy_t block_null_copy_func,
                           __in__ unsigned scan_batch_size,
                           __in__ unsigned filter_batch_size,
                            __in__ bool filter_hint_expected_avg_batch_eval_is_non_empty) :
                        super(nullptr, scan_batch_size), tuplet_ids_interval_begin(tuplet_ids_interval_begin),
                        tuplet_ids_interval_end(tuplet_ids_interval_end), block_copy_func(block_copy_func),
                        block_null_copy_func(block_null_copy_func)
                {
                    assert (tuplet_ids_interval_begin != nullptr && tuplet_ids_interval_end != nullptr);
                    assert (tuplet_ids_interval_begin < tuplet_ids_interval_end);
                    filter_operator = new filter_t(destination, predicate, null_policy,
                                                   filter_batch_size,
                                                   filter_hint_expected_avg_batch_eval_is_non_empty);
                    super::add_destination(filter_operator);
                }

                virtual void on_start() override
                {
                    assert (filter_operator != nullptr);
                    debug_create_variable(size_t, last_upperbound, 0);

                    auto interval = tuplet_ids_interval_begin;
                    auto count = (tuplet_ids_interval_end - tuplet_ids_interval_begin);
                    while (count--) {
                        debug_exec(
                            assert (interval->get_type() == interval_t::bounds_policy::right_open);
                            assert (interval->get_lower_bound() >= last_upperbound);
                            last_upperbound = interval->get_upper_bound();
                        );
                        super::produce_tupletid_range(interval->get_lower_bound(), interval->get_upper_bound(),
                                                      block_copy_func, block_null_copy_func);
                        ++interval;
                    }
                }

                virtual void on_cleanup() override
                {
                    assert_non_null (filter_operator);
                    delete filter_operator;
                    filter_operator = nullptr;
                }

                virtual const filter_t *get_filter() const final
                {
                    assert_non_null (filter_operator);
                    return filter_operator;
                }

                virtual const char *get_class_name() const override
                {
                    return "vpipes::pipes::table_scan";
                }
            };
        }
    }
}
