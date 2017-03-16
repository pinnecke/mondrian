#pragma once

#include "producer.hpp"
#include "consumer.hpp"

namespace mondrian
{
    namespace vpipes
    {
        template <class Type = size_t>
        class interval
        {
        public:
            using type_t = Type;

            enum class bounds_policy { right_open, left_open, open, closed };

        private:
            type_t lower_bound, upper_bound;
            bounds_policy type;

        public:
            interval (type_t lower_bound, type_t upper_bound, bounds_policy type = bounds_policy::right_open):
                    lower_bound(lower_bound), upper_bound(upper_bound), type(type)
            {
                assert (lower_bound <= upper_bound);
            }

            type_t get_lower_bound() const
            {
                return lower_bound;
            }

            type_t get_upper_bound() const
            {
                return upper_bound;
            }

            bounds_policy get_type() const
            {
                return type;
            }

            size_t get_distance() const
            {
                auto start = lower_bound + ((type == bounds_policy::left_open) || (type == bounds_policy::open)? 1 : 0);
                auto end   = upper_bound + ((type == bounds_policy::right_open) || (type == bounds_policy::open)? 1 : 0);
                return (end - start);
            }


        };

    }
}