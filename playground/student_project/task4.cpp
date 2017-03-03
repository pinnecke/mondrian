#include <iostream>
#include <vpipes.hpp>
#include "tasks.hpp"


using namespace std;
using namespace mondrian::vpipes;

int main() {
    size_t num_elements = 2e6;
    size_t vector_size = 1000;
    auto column = create_column(num_elements);

    auto print = toolkit::printer<int>();
    auto read = toolkit::reader<int>(&print, column, column + num_elements, vector_size);
    read.start();

    delete_column(column);

    return EXIT_SUCCESS;
}

