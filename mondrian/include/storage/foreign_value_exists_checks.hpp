#ifndef PANTHEON_FOREIGN_VALUE_EXISTS_CHECKS_HPP
#define PANTHEON_FOREIGN_VALUE_EXISTS_CHECKS_HPP

namespace pantheon
{
    namespace storage
    {
        namespace foreign_value_exists_checks
        {
            template <class ValueType>
            using function_t = function<bool(const ValueType *value, size_t num_of_values)>;

            template <class ValueType>
            struct any_value_exists
            {
                bool operator()(const ValueType *value, size_t num_of_values) {
                    return true;
                }
            };
        }
    }
}

#endif //PANTHEON_FOREIGN_VALUE_EXISTS_CHECKS_HPP
