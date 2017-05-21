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
                mtl::smart_array<size_t> batch_operator_matching_indicies;
                mtl::smart_array<size_t> column_operator_matching_indicies;
                size_t indicies_offset;
                statistics::join_run statistics;
                which_operator curr_op;
                map<output_t, vector<input_t > > join_map;
                join_func_t join_condition;

            public:
                block_nested_join(__in__ consumer_t *destination,
                       __in__ unsigned batch_size,
                       __in__ join_func_t join_condition) :
                        super(destination, batch_size),
                        batch_operator_matching_indicies(batch_size),
                        column_operator_matching_indicies(batch_size),
                        join_condition(join_condition),
                        indicies_offset(0)
                {
                    curr_op = which_operator::neither;
                }

                inline virtual void on_consume(__in__ const input_batch_t *data,
                                               __in__ which_operator operator_role) override final __attribute__((always_inline))
                {
                    if (curr_op == which_operator::neither) {
                        curr_op = operator_role;
                    }

                    if (curr_op == operator_role) {
                        build_join_table(data, data->get_null_mask());
                    } else {
//                        batch_operator_matching_indicies.resize(join_map.size() * data->get_size() *100);
//                        column_operator_matching_indicies.resize(join_map.size() * data->get_size() * 100);

                        size_t num_of_matchs = 0;
                        join_tables(num_of_matchs, data, data->get_null_mask());

                        super::produce(batch_operator_matching_indicies.get_content(),
                                       column_operator_matching_indicies.get_content(), data->get_null_mask(),
                                       num_of_matchs, false);
                    }
                }

                virtual void on_cleanup() override
                {
                    super::close();
                }

                void clean_storage()
                {
                    batch_operator_matching_indicies.dispose();
                    column_operator_matching_indicies.dispose();
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
                void build_join_table(__in__ const input_batch_t *data,
                                      __in__ const mtl::smart_bitmask *null_mask) __attribute__((always_inline))
                {
                        auto batch_size = data->get_size();
                        auto data_ids = data->get_tupletids();
                        auto data_values = data->get_values();

                        if (null_mask->is_unset()) {
                            ++this->statistics.count_non_null_branch_used;

                            for (auto i = 0; i < batch_size; ++i) {
                                join_map[data_values[i]].push_back(data_ids[i]);
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

                                join_map[data_values[i]].push_back(data_ids[i]);
                            }

                        }
                }

                void join_tables(__out__ size_t &num_of_matchs,
                                 __in__ const input_batch_t *data,
                                 __in__ const mtl::smart_bitmask *null_mask) __attribute__((always_inline))
                {
                    auto batch_size = data->get_size();
                    auto data_ids = data->get_tupletids();
                    auto data_values = data->get_values();
                    auto offset =0 ;

                    if (null_mask->is_unset()) {

                        ++this->statistics.count_non_null_branch_used;

                        for (auto i = 0; i < batch_size; ++i) {

                            for (auto const& imap : join_map) {
                                if (join_condition (data_values[i], imap.first)) {
                                    auto num_ids = join_map[imap.first].size();
                                    num_of_matchs += num_ids;
                                    this->statistics.count_join_pairs += num_ids;

                                    for (auto j = 0; j < num_ids; ++j) {
                                        batch_operator_matching_indicies.set(j+offset,data_ids[i]);
                                        column_operator_matching_indicies.set(j+offset,join_map[imap.first].at(j));
                                    }

                                    offset += num_ids;
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

                           for (auto const& imap : join_map) {

                                if (join_condition (data_values[i], imap.first)) {
                                    auto num_ids = join_map[imap.first].size();
                                    num_of_matchs += num_ids;
                                    this->statistics.count_join_pairs += num_ids;
                                    for (auto j = 0; j < num_ids; ++j) {
                                        batch_operator_matching_indicies.set(j+offset,data_ids[i]);
                                        column_operator_matching_indicies.set(j+offset,join_map[imap.first].at(j));
                                    }

                                    offset += num_ids;
                                }
                            }

                        }
                    }
                }
            };
        }
    }
}

