#pragma once

#include <functional>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <exception>

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
#include "vpipes/pipe_head.hpp"
#include "vpipes/datastructrues/datastructure.hpp"
#include "vpipes/datastructrues/red_black_tree.hpp"

#include "vpipes/pipes/filter.hpp"
#include "vpipes/pipes/materializer.hpp"
#include "vpipes/pipes/materializers/val_materialize.hpp"
#include "vpipes/pipes/materializers/tid_materialize.hpp"
#include "vpipes/pipes/no_operation.hpp"
#include "vpipes/pipes/table_scan.hpp"
#include "vpipes/pipes/map.hpp"
#include "vpipes/pipes/tee.hpp"
#include "vpipes/pipes/project.hpp"
#include "vpipes/pipes/dedup.hpp"