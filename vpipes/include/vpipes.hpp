#pragma once

#include <functional>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <exception>
#include <stddef.h>

#include "../../mtl/include/mtl"

#ifndef NDEBUG
#define assert_non_null(x) \
{                          \
    assert (x != nullptr); \
}
#else
#define assert_non_null(x)  \
    { }
#endif

#define __in__
#define __out__

namespace mondrian
{
    namespace vpipes
    {
        enum class null_info { contains_null, non_null };

        namespace statistics
        {
            struct predicate_run
            {
                size_t count_null_branch_used = 0, count_non_null_branch_used = 0;
                size_t count_null_values = 0;
                size_t count_satisfying_values = 0, count_non_satisfying_values = 0, count_skipped_values = 0;
            };

            struct operator_run
            {
                size_t num_batches = 0;
                size_t num_tuplets = 0;
                size_t num_empty_batches = 0;
            };

            struct join_run
            {
                size_t count_null_branch_used = 0, count_non_null_branch_used = 0;
                size_t count_null_values = 0;
                size_t count_join_pairs = 0;
            };

        }

        using tuplet_id_t = size_t;

        enum class null_value_filter_policy { skip_null_values, remove_null_values };
    }
}

#include "vpipes/macros.hpp"
#include "vpipes/iterator.hpp"
#include "vpipes/interval.hpp"
#include "vpipes/memory.hpp"
#include "vpipes/predicate_func.hpp"
#include "vpipes/map_func.hpp"
#include "vpipes/batch.hpp"
#include "vpipes/consumer.hpp"
#include "vpipes/producer.hpp"
#include "vpipes/pipe.hpp"
#include "vpipes/bi_consumer.hpp"
#include "vpipes/bi_pipe.hpp"
#include "vpipes/join_func.hpp"
#include "vpipes/pipes/joins/hash_libs/libcuckoo/cuckoohash_map.hh"

#include "vpipes/pipes/filter.hpp"
#include "vpipes/pipes/materializer.hpp"
#include "vpipes/pipes/materializers/val_materialize.hpp"
#include "vpipes/pipes/materializers/tid_materialize.hpp"
#include "vpipes/pipes/materializers/nullmask_materialize.hpp"
#include "vpipes/pipes/no_operation.hpp"
#include "vpipes/pipes/table_scan.hpp"
#include "vpipes/pipes/map.hpp"
#include "vpipes/pipes/tee.hpp"
#include "vpipes/pipes/attribute_switch.hpp"
#include "vpipes/pipes/joins/block_nested_join.hpp"
#include "vpipes/pipes/joins/stl_hash_join.hpp.hpp"
#include "vpipes/pipes/joins/libcuckoo_hash_join.hpp"
#include "vpipes/pipes/joins/dense_map_hash_join.hpp"
#include "vpipes/pipes/joins/sparse_map_hash_join.hpp"