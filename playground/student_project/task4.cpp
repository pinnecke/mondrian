#include <iostream>
#include "framework/pipe_heads/reader.hpp"
#include "framework/pipes/filters.hpp"
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
    auto read = reader<int>(&print, column, column + num_elements, vector_size);
    read.start();

    delete_column(column);

    return EXIT_SUCCESS;
}

