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

#include "../vpipes.hpp"

namespace mondrian
{
    namespace vpipes
    {
        template <class ValueType>
        struct point_copy
        {
            using value_t = ValueType;
            using func_t = std::function<void(value_t *out, const tuplet_id_t *tupletids, size_t num_of_ids)>;
        };

        template <class ValueType>
        struct block_copy
        {
            using value_t = ValueType;
            using func_t = std::function<void(value_t *out, tuplet_id_t begin, tuplet_id_t end)>;
        };

        struct point_null_copy
        {
            using func_t = std::function<void(mtl::smart_bitmask *out, const tuplet_id_t *tupletids, size_t num_of_ids)>;
        };

        struct block_null_copy
        {
            using func_t = std::function<void(mtl::smart_bitmask *out, tuplet_id_t begin, tuplet_id_t end)>;
        };
    }
}

