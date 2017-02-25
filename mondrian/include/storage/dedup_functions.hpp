#ifndef PANTHEON_DEDUP_FUNCTIONS_HPP
#define PANTHEON_DEDUP_FUNCTIONS_HPP

namespace pantheon
{
    namespace storage
    {
        namespace dedup_functions
        {
            template <class ValueType>
            struct sort_based
            {
                ValueType *operator()(size_t *num_of_dedup_values, vector<bool> *null_mask_out,
                                      bool *free_on_retval_required,
                                      ValueOrderPolicy value_order_policy,
                                      const ValueType *contained_values, size_t num_of_contained_values,
                                      Trilean contained_values_are_duplicate_free,
                                      bool contained_values_are_nullable,
                                      const vector<bool> *contained_values_null_mask,
                                      const ValueType *new_values, size_t num_of_new_values,
                                      bool new_values_are_nullable,
                                      const vector<bool> *new_values_null_mask,
                                      function<bool(const ValueType lhs, const ValueType rhs)> comp,
                                      NullDuplicateHandlingPolicy null_handling)
                {
                    assert (num_of_dedup_values != nullptr);
                    assert (!new_values_are_nullable || null_mask_out != nullptr);
                    assert (free_on_retval_required != nullptr);
                    assert (num_of_contained_values == 0 || contained_values != nullptr);
                    assert (!contained_values_are_nullable || contained_values_null_mask != nullptr);
                    assert (new_values != nullptr);

                    using namespace pantheon::collections::operations;

                    size_t num_final_deduped_values = 0;
                    vector<bool> new_values_deduped_null_mask;

                    if (num_of_new_values == 0 || new_values == nullptr) {
                        *free_on_retval_required = false;
                        return nullptr;
                    } else {

                        // Dedup new values, only
                        ValueType *new_values_deduped = (ValueType *) malloc (num_of_new_values * sizeof(ValueType));
                        ValueType *new_values_deduped_end;

                        assert (new_values_deduped != nullptr);

                        if (new_values_are_nullable) {
                            new_values_deduped_null_mask.reserve(num_of_new_values);

                            /*   for (const ValueType *lhs = new_values; lhs != new_values + num_of_new_values; ++lhs) {
                                   for (const ValueType *rhs = new_values_deduped; rhs != new_values_deduped + num_new_values_deduped; ++rhs) {
                                       if (lhs != rhs) {
                                           if (new_values_null_mask->at(lhs - new_values)) {
                                               if (new_values_deduped_null_mask[rhs - new_values_deduped]) {
                                                   // the value for lhs is NULL and the value for rhs is NULL
                                                   if (null_handling == NullDuplicateHandlingPolicy::AllowDuplicateNullValues) {
                                                       new_values_deduped_null_mask[num_new_values_deduped++] = true;
                                                   }
                                               } else {
                                                   // the value for lhs is NULL and the value for rhs is non-NULL
                                                   continue;
                                               }
                                           } else {
                                               if (new_values_deduped_null_mask[rhs - new_values_deduped]) {
                                                   // the value for lhs is non-NULL and the value for rhs is NULL
                                                   new_values_deduped[num_new_values_deduped] = *lhs;
                                                   new_values_deduped_null_mask[num_new_values_deduped++] = true;
                                               } else {
                                                   // the value for lhs is non-NULL and the value for rhs is non-NULL
                                                   if (*lhs != *rhs) {
                                                       new_values_deduped[num_new_values_deduped] = *lhs;
                                                       new_values_deduped_null_mask[num_new_values_deduped++] = false;
                                                   }
                                               }
                                           }
                                       }
                                   }
                               }*/
                        } else {
                            //if (value_order_policy == ValueOrderPolicy::Instable) {
                            new_values_deduped_end = dedup::sort_based<ValueType>(new_values, new_values + num_of_new_values, new_values_deduped, comp);
                            // } else {
                            // TODO:...
                            // }

                            std::for_each(new_values_deduped, new_values_deduped_end, [] (auto x) { std::cout << x << std::endl; });
                        }

                        // TODO: set difference

                        // Dedup new values considering existing values
                        size_t num_new_values_deduped = new_values_deduped_end - new_values_deduped;
                        ValueType *final_input_values = (ValueType *) malloc (num_new_values_deduped * sizeof(ValueType));
                        assert (final_input_values != nullptr);

                        /*
                        for (const ValueType *lhs = new_values_deduped; lhs != new_values_deduped + num_new_values_deduped; ++lhs) {
                            for (const ValueType *rhs = contained_values; rhs != contained_values + num_of_contained_values; ++rhs) {
                                if (lhs != rhs && *lhs != *rhs) {
                                    final_input_values[num_final_deduped_values++] = *lhs;
                                }
                            }
                        }*/

                        // free(new_values_deduped);

                        *num_of_dedup_values = num_final_deduped_values;
                        *free_on_retval_required = true;

                        return final_input_values;
                    }
                }
            };
        }
    }
}

#endif //PANTHEON_DEDUP_FUNCTIONS_HPP
