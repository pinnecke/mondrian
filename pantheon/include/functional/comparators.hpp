#ifndef PANTHEON_COMPARATORS_HPP
#define PANTHEON_COMPARATORS_HPP

namespace pantheon
{
    namespace functional
    {
        namespace comparators
        {
            template <class ValueType>
            using function_t = function<bool(const ValueType lhs, const ValueType rhs)>;

            template <class ValueType>
            struct less
            {
                bool operator()(const ValueType lhs, const ValueType rhs) const {
                    return lhs < rhs;
                }
            };
        }
    }
}

#endif //PANTHEON_COMPARATORS_HPP
