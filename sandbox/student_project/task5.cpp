#include <iostream>
#include "framework/pipe_heads/reader.hpp"
#include "framework/pipes/sequential_filter.hpp"
#include "framework/pipe_tails/printer.hpp"
#include "tasks.hpp"

using namespace std;
using namespace mondrian::query_engine::operators::sinks;
using namespace mondrian::query_engine::operators::sources;
using namespace mondrian::query_engine::operators::sql;

int main() {
    size_t num_elements = 2e6;
    size_t vector_size = 1000;
    auto column = create_column(num_elements);

    auto print = printer<int>();
    auto filter = sequential_filter<int>(&print, vector_size, [] (const int *x) { return *x >= 0; });
    auto read = reader<int>(&filter, column, column + num_elements, vector_size);
    read.produce();

    delete_column(column);

    return EXIT_SUCCESS;
}

