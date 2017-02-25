#ifndef PANTHEON_DEFAULT_VALUE_SUPPLIERS_HPP
#define PANTHEON_DEFAULT_VALUE_SUPPLIERS_HPP

namespace pantheon
{
    namespace storage
    {
        namespace default_value_suppliers
        {
            template <class ValueType>
            using function_t = function<ValueType()>;

            template <class ValueType>
            struct default_constructor
            {
                ValueType operator()() {
                    return ValueType();
                }
            };
        }
    }
}

#endif //PANTHEON_DEFAULT_VALUE_SUPPLIERS_HPP
