#include <iostream>

#include "tasks.hpp"

using namespace std;

//

int main() {
    size_t num_elements = 2e6;
    int *column = create_column(num_elements);

    for (size_t i = 0; i < num_elements; ++i)
        if (column[i] != *(column + i))
            return 1;

    for (int *it = column; it < column + num_elements; ++it)
        if (column[(it - column)] != *it)
            return 2;

    int *end = column, *begin = column;
    advance(end, num_elements);
    for (auto it = begin; it != end; ++it)
        if (&column[distance(begin, it)] != it)
            return 3;

    delete_column(column);
    return 0;
}

