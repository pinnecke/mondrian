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
#include "vpipes/storage.hpp"
#include "vpipes/predicate_func.hpp"
#include "vpipes/map_func.hpp"
#include "vpipes/batch.hpp"
#include "vpipes/consumer.hpp"
#include "vpipes/producer.hpp"
#include "vpipes/pipe.hpp"
#include "vpipes/pipe_head.hpp"

#include "vpipes/pipes/filter.hpp"
#include "vpipes/pipes/materialize.hpp"
#include "vpipes/pipes/no_operation.hpp"
#include "vpipes/pipes/table_scan.hpp"