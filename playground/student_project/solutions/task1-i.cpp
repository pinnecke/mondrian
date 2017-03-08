#include <iostream>
#include <limits>
#include <random>

using namespace std;

// To answer the question; one has just to execute the program. It returns: 0
// Reason:
//          (column[i] != *(column + i))            Value-compare: (the i-th value in column) and (the address of the
//                                                                 column incremented by i elements + dereferenced =
//                                                                 the i-th value in column)
//                                                  Result: Always true, hence, ro return of 1
//
//          (column[(it - column)] != *it)          Value-compare: (Left-hand-side is i-th value in the column where
//                                                                 i is computed from the distance between the address
//                                                                 of the current iterator to the starting address) and
//                                                                 (dereferencing the value behind the address of the
//                                                                  current iterator)
//                                                  Result: Always true, hence, ro return of 2
//
//          (&column[distance(begin, it)] != it)    Pointer-compare: (first the i-th index of the element in column is
//                                                                  computed (the STL-styled way), and than the
//                                                                  address of that value is taken) and the address
//                                                                  of the current iterator.
//                                                  Result: Always true (same addresses), hence, ro return of 3

default_random_engine generator;

int random_number()
{
    static uniform_int_distribution<int> distribution(numeric_limits<int>::min(), numeric_limits<int>::max());
    return distribution(generator);
};

int *create_column(unsigned long num_of_elements, bool fill_with_random = true, bool fill = true)
{
    auto result = (int *) malloc (num_of_elements * sizeof(int));
    if (fill) {
        for (auto i = 0; i < num_of_elements; ++i)
            result[i] = fill_with_random ? random_number() : i;
    }
    return result;
}

void delete_column(int *column)
{
    free (column);
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

