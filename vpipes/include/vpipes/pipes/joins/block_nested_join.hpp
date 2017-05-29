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
#include <map>
#include <vector>

using namespace std;

namespace mondrian
{
    namespace vpipes
    {
        namespace bi_pipes
        {
            template<class Input>
            class block_nested_join : public bi_pipe<Input, Input>
            {
                using super = bi_pipe<Input, Input>;

            public:
                using typename super::input_t;
                using typename super::input_batch_t;
                using typename super::output_t;
                using typename super::output_batch_t;
                using typename super::consumer_t;
                using join_func_t = typename vpipes::predicates::join_conditions<input_t>::func_t;

            private:
                mtl::smart_array<size_t> outer_operand_matching_indicies;
                mtl::smart_array<size_t> inner_operand_matching_indicies;
                statistics::join_run statistics;
                which_operand curr_op;
                mtl::smart_array<size_t> inner_operand_ids;
                mtl::smart_array<size_t> inner_operand_values;
                join_func_t join_condition;

            public:
                block_nested_join(__in__ consumer_t *destination,
                       __in__ unsigned batch_size,
                       __in__ join_func_t join_condition) :
                        super(destination, batch_size),
                        outer_operand_matching_indicies(batch_size),
                        inner_operand_matching_indicies(batch_size),
                        inner_operand_ids(batch_size),
                        inner_operand_values(batch_size),
                        join_condition(join_condition)
                {
                    curr_op = which_operand::neither;
                }

                inline virtual void on_consume(__in__ const input_batch_t *data,
                                               __in__ which_operand operand_role) override final __attribute__((always_inline))
                {
                    if (curr_op == which_operand::neither) {
                        curr_op = operand_role;
                    }

                    if (curr_op == operand_role) {
                        store_inner_tuples(data, data->get_null_mask());
                    } else {
                        size_t num_of_matchs = 0;
                        find_matches(num_of_matchs, data, data->get_null_mask());

                        super::produce(outer_operand_matching_indicies.get_content(),
                                       inner_operand_matching_indicies.get_content(), data->get_null_mask(),
                                       num_of_matchs, false);
                    }
                }

                virtual void on_cleanup() override
                {
                    super::close();
                }

                void clean_storage()
                {
                    outer_operand_matching_indicies.dispose();
                    inner_operand_matching_indicies.dispose();
                    inner_operand_ids.dispose();
                    inner_operand_values.dispose();
                }

                const statistics::join_run *get_join_statistics() const
                {
                    return &this->statistics;
                }

                virtual const char *get_class_name() const override
                {
                    return "vpipes::pipes::block_nested_join";
                }

            private:
                void store_inner_tuples(__in__ const input_batch_t *data,
                                      __in__ const mtl::smart_bitmask *null_mask) __attribute__((always_inline))
                {
                        auto batch_size = data->get_size();
                        auto data_ids = data->get_tupletids();
                        auto data_values = data->get_values();

                        if (null_mask->is_unset()) {
                            ++this->statistics.count_non_null_branch_used;

                            for (auto i = 0; i < batch_size; ++i) {
                                  inner_operand_ids.add(data_ids[i]);
                                  inner_operand_values.add(data_values[i]);
                            }
                        } else {
                            ++this->statistics.count_null_branch_used;
                            bool tuplet_is_null;

                            for (size_t i = 0; i < batch_size; ++i) {
                                SMART_BITMASK_GET_UNSAFE_FAST(tuplet_is_null, null_mask, i);

                                if (tuplet_is_null) {
                                    ++this->statistics.count_null_values;
                                    continue;
                                }

                                inner_operand_ids.add(data_ids[i]);
                                inner_operand_values.add(data_values[i]);
                            }

                        }
                }

                void find_matches(__out__ size_t &num_of_matchs,
                                 __in__ const input_batch_t *data,
                                 __in__ const mtl::smart_bitmask *null_mask) __attribute__((always_inline))
                {
                    auto batch_size = data->get_size();
                    auto data_ids = data->get_tupletids();
                    auto data_values = data->get_values();
                    num_of_matchs = 0;
                    if (null_mask->is_unset()) {

                        ++this->statistics.count_non_null_branch_used;

                        for (auto i = 0; i < batch_size; ++i) {
                            auto size_of_inner_table = inner_operand_ids.get_num_elements();
                            auto inner_table_raw_vals = inner_operand_values.get_raw_data();
                            auto inner_table_raw_ids = inner_operand_ids.get_raw_data();
                            for (auto j = 0; j < size_of_inner_table; ++j) {
                                if (join_condition (data_values[i], inner_table_raw_vals[j])) {
                                    ++this->statistics.count_join_pairs;
                                    outer_operand_matching_indicies.set(num_of_matchs,data_ids[i]);
                                    inner_operand_matching_indicies.set(num_of_matchs,inner_table_raw_ids[j]);
                                    ++num_of_matchs;
                                }
                            }
                        }
                    } else {
                        ++this->statistics.count_null_branch_used;
                        bool tuplet_is_null;

                        for (size_t i = 0; i < batch_size; ++i) {
                            SMART_BITMASK_GET_UNSAFE_FAST(tuplet_is_null, null_mask, i);

                            if (tuplet_is_null) {
                                ++this->statistics.count_null_values;
                                continue;
                            }

                            auto size_of_inner_table = inner_operand_ids.get_num_elements();
                            auto inner_table_raw_vals = inner_operand_values.get_raw_data();
                            auto inner_table_raw_ids = inner_operand_ids.get_raw_data();
                            for (auto j = 0; j < size_of_inner_table; ++j) {
                                if (join_condition (data_values[i], inner_table_raw_vals[j])) {
                                    ++this->statistics.count_join_pairs;
                                    outer_operand_matching_indicies.set(num_of_matchs,data_ids[i]);
                                    inner_operand_matching_indicies.set(num_of_matchs,inner_table_raw_ids[j]);
                                    ++num_of_matchs;
                                }
                            }
                       }
                    }
                }
            };
        }
    }
}

