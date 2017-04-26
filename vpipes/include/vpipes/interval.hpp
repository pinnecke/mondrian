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
            interval (__in__ type_t lower_bound,
                      __in__ type_t upper_bound,
                      __in__ bounds_policy type = bounds_policy::right_open):
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