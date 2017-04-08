//
// Created by Mahmoud Mohsen on 4/5/17.
//

#pragma once

#include <vpipes.hpp>


namespace mondrian {

    namespace vpipes {

        namespace pipes {

            template<class Input, class InputTupletIdType = size_t>
            class dedup_1_flavour : public pipe<Input, Input, InputTupletIdType, InputTupletIdType>
            {
                using super = pipe<Input, Input, InputTupletIdType, InputTupletIdType>;
                using data_struct_ids_t = mondrian::vpipes::datastructures::datastructure<InputTupletIdType>;
                using data_struct_vals_t = mondrian::vpipes::datastructures::datastructure<Input>;
            public:
                using typename super::input_t;
                using typename super::input_tupletid_t;
                using typename super::input_batch_t;
                using typename super::output_t;
                using typename super::output_tupletid_t;
                using typename super::output_batch_t;
                using typename super::consumer_t;

                enum class policy
                {
                    Ids, values, both
                };

            private:
                size_t *matching_indices_buffer;
                size_t buffer_size;
                bool hint_avg_batch_eval_result_is_non_empty;
                data_struct_ids_t *m_ids_data_struct;
                data_struct_vals_t *m_vals_data_struct;
                policy m_policy;

                template<class Container, class DataStruct>
                inline void dedup_container(size_t  *out_matching_indices, size_t *out_num_matching_indices,
                                            Container *container, DataStruct *Ds,
                                            size_t num_elements)
                {
                    const size_t *out_matching_indices_start = out_matching_indices;
                    for (auto idx = 0; idx < num_elements; ++idx) {
                        if (!Ds->contains(container + idx)) {
                            *out_matching_indices++ = idx;
                            Ds->insert(container + idx);
                        }
                    }
                    *out_num_matching_indices = (out_matching_indices - out_matching_indices_start);
                }

                template<class Container1, class Container2, class DataStruct1, class DataStruct2>
                inline void dedup_2containers(size_t *out_matching_indices, size_t *out_num_matching_indices,
                                              Container1 *container1, Container2 *container2, DataStruct1 *Ds1,
                                              DataStruct2 *Ds2,
                                              size_t num_elements)
                {
                    size_t result_size = 0;
                    for (auto idx = 0; idx != num_elements; ++idx) {
                        if (!(Ds1->contains(container1 + idx) && Ds2->contains(container2 + idx))) {
                            out_matching_indices[idx] = idx;
                            Ds1->insert(container1 + idx);
                            Ds2->insert(container2 + idx);
                            ++result_size;
                        }
                    }
                    *out_num_matching_indices = result_size;
                }


            public:dedup_1_flavour(consumer_t *destination, unsigned batch_size, data_struct_ids_t *data_struct_ids_p,
                                   data_struct_vals_t *data_struct_vals_p, policy policy_p,
                                bool hint_avg_batch_eval_result_is_non_empty) :
                        super(destination, batch_size), m_ids_data_struct(data_struct_ids_p),
                        m_vals_data_struct(data_struct_vals_p), m_policy(policy_p),
                        hint_avg_batch_eval_result_is_non_empty(hint_avg_batch_eval_result_is_non_empty)
                {
                    // Note here: The operator is unaware of the batch size of the input. The assignment
                    // of the batch size of this operator as the batch size of the preceding operator
                    // just a best guess and must be corrected afterwards if it was wrong
                    buffer_size = batch_size;
                    matching_indices_buffer = (size_t *) malloc(buffer_size * sizeof(size_t));
                    assert(matching_indices_buffer != nullptr);
                }

                inline virtual void on_consume(const input_batch_t *data) override final __attribute__((always_inline)) {

                    auto input_batch_size = data->get_size();
                    if (__builtin_expect(input_batch_size > buffer_size, false))
                    {
                        buffer_size = input_batch_size;
                        matching_indices_buffer = (size_t *) realloc(matching_indices_buffer, input_batch_size *
                                                                                              sizeof(input_tupletid_t));

                        assert(matching_indices_buffer != nullptr);
                    }

                    size_t result_size = 0;

                    if (m_policy == policy::values)
                    {
                        dedup_container<Input, data_struct_vals_t>(matching_indices_buffer, &result_size,
                                                                   data->get_values_begin(), m_vals_data_struct,
                                                                   input_batch_size);

                    } else if (m_policy == policy::Ids)
                    {
                        dedup_container<InputTupletIdType, data_struct_ids_t>(matching_indices_buffer, &result_size,
                                                                              data->get_tupletids_begin(),
                                                                              m_ids_data_struct, input_batch_size);
                    } else
                    {
                        dedup_2containers<Input, InputTupletIdType, data_struct_vals_t, data_struct_ids_t>(
                                matching_indices_buffer, &result_size, data->get_values_begin(),
                                data->get_tupletids_begin(), m_vals_data_struct, m_ids_data_struct, input_batch_size);
                    }
                    assert(result_size <= buffer_size);
                    if (__builtin_expect(result_size != 0, hint_avg_batch_eval_result_is_non_empty))
                    {
                        super::produce(data->get_tupletids_begin(), data->get_values_begin(), matching_indices_buffer,
                                       result_size, false);
                    }

                }


                virtual void on_cleanup() override
                {
                    free(matching_indices_buffer);
                }

            };

        }

    }
}
