#pragma once

#include <functional>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <exception>

#include "vpipes/macros.hpp"
#include "vpipes/iterator.hpp"
#include "vpipes/chunk.hpp"
#include "vpipes/functional.hpp"
#include "vpipes/consumer.hpp"
#include "vpipes/batch_pipe.hpp"
#include "vpipes/bi_consumer.hpp"
#include "vpipes/producer.hpp"
#include "vpipes/pipe.hpp"
#include "vpipes/bi_pipe.hpp"
#include "vpipes/pipe_head.hpp"

#include "vpipes/interval.hpp"
#include "vpipes/toolkit/collector.hpp"
#include "vpipes/toolkit/filters.hpp"
#include "vpipes/toolkit/materialize.hpp"
#include "vpipes/toolkit/printer.hpp"
#include "vpipes/toolkit/reader.hpp"
#include "vpipes/toolkit/table_scan.hpp"
#include "vpipes/toolkit/test_bi_pipe.hpp"