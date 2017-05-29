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
#include <unordered_map>
#include <vector>
#include <sparsehash/dense_hash_map>

using namespace std;

namespace mondrian
{
    namespace vpipes
    {
        namespace bi_pipes
        {
            template<class Input>
            class dense_map_hash_join : public bi_pipe<Input, Input>
            {
                using super = bi_pipe<Input, Input>;

            public:
                using typename super::input_t;
                using typename super::input_batch_t;
                using typename super::output_t;
                using typename super::output_batch_t;
                using typename super::consumer_t;

            private:
                mtl::smart_array<size_t> outer_operand_matching_indicies;
                mtl::smart_array<size_t> inner_operand_matching_indicies;
                statistics::join_run statistics;
                which_operand curr_op;
                google::dense_hash_map <output_t, vector<input_t> > join_map;

            public:
                dense_map_hash_join (__in__ consumer_t *destination,
                                     __in__ unsigned batch_size,
                                     __in__ output_t dense_map_empty_key) :
                        super(destination, batch_size),
                        outer_operand_matching_indicies(batch_size),
                        inner_operand_matching_indicies(batch_size)
                {
                    curr_op = which_operand::neither;
                    join_map.set_empty_key(dense_map_empty_key);
                }

                inline virtual void on_consume(__in__ const input_batch_t *data,
                                               __in__ which_operand operand_role) override final __attribute__((always_inline))
                {
                    if (curr_op == which_operand::neither) {
                        curr_op = operand_role;
                    }

                    if (curr_op == operand_role) {
                        hash_build(data, data->get_null_mask());
                    } else {

                        size_t num_of_matchs = 0;
                        hash_probe(num_of_matchs, data, data->get_null_mask());

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
                    join_map.clear();
                }

                const statistics::join_run *get_join_statistics() const
                {
                    return &this->statistics;
                }

                virtual const char *get_class_name() const override
                {
                    return "vpipes::pipes::dense_map_hash_join";
                }

            private:
                void hash_build(__in__ const input_batch_t *data,
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

                void hash_probe(__out__ size_t &num_of_matchs,
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

                            if (join_map.find(data_values[i]) == join_map.end())
                                continue;

                            auto num_ids = join_map[data_values[i]].size();
                            num_of_matchs += num_ids;
                            this->statistics.count_join_pairs += num_ids;

                            for (auto j = 0; j < num_ids; ++j) {
                                outer_operand_matching_indicies.set(offset + j,data_ids[i]);
                                inner_operand_matching_indicies.set(offset + j,join_map[data_values[i]].at(j));
                            }
                            offset += num_ids;
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

                            if (join_map.find(data_values[i]) == join_map.end())
                                continue;

                            auto num_ids = join_map[data_values[i]].size();
                            num_of_matchs += num_ids;
                            this->statistics.count_join_pairs += num_ids;

                            for (auto j = 0; j < num_ids; ++j) {
                                outer_operand_matching_indicies.set(offset + j,data_ids[i]);
                                inner_operand_matching_indicies.set(offset + j,join_map[data_values[i]].at(j));
                            }
                            offset += num_ids;
                        }
                    }
                }
            };
        }
    }
}

