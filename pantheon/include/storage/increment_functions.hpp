#ifndef PANTHEON_INCREMENT_FUNCTIONS_HPP
#define PANTHEON_INCREMENT_FUNCTIONS_HPP

namespace pantheon
{
    namespace storage
    {
        namespace increment_functions
        {
            template <class ValueType>
            using function_t = function<ValueType(ValueType *value)>;

            template <class ValueType>
            struct numeric_increment
            {
                ValueType operator()(ValueType *value) {
                    return (*value)++;
                }
            };
        }
    }
}

#endif //PANTHEON_INCREMENT_FUNCTIONS_HPP
