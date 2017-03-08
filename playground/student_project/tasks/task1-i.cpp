#include <iostream>

using namespace std;

int random_number()
{
    // TODO: Add your code here
};

int *create_column(unsigned long num_of_elements, bool fill_with_random = true, bool fill = true)
{
    // TODO: Add your code here
}

void delete_column(int *column)
{
    // TODO: Add your code here
}


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

