#ifndef PANTHEON_VALUE_CONSTRAINTS_HPP
#define PANTHEON_VALUE_CONSTRAINTS_HPP

namespace pantheon
{
    namespace storage
    {
        namespace value_constraints
        {
            template <class ValueType>
            using function_t = function<char *(bool *all_satisfy, const ValueType *values_contained, size_t num_of_values, bool values_might_be_null,
                                               const ValueType *values_to_be_added, size_t num_of_values_to_be_added,
                                               const vector<bool> *values_contained_null_mask)>;

            template <class ValueType>
            struct no_constraints
            {
                char *operator()(bool *all_satisfy, const ValueType *values_contained, size_t num_of_values, bool values_might_be_null,
                                 const ValueType *values_to_be_added, size_t num_of_values_to_be_added,
                                 const vector<bool> *values_contained_null_mask) {
                    if (num_of_values == 0) {
                        assert (all_satisfy != nullptr);
                        *all_satisfy = true;
                        return nullptr;
                    } else {
                        *all_satisfy = false;
                        char *m = (char *) malloc(100 * sizeof(char));
                        strcpy(m, "Too many ;D");
                        return m;
                    }
                }
            };
        }
    }
}

#endif //PANTHEON_VALUE_CONSTRAINTS_HPP
