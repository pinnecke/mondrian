#include <iostream>
#include <vpipes.hpp>
#include <random>
#include "profiling.hpp"

using namespace std;
using namespace mondrian::vpipes;

#define PREDICATE(x)                                                                    \
[] (int ***result, size_t *result_size, int **begin, int **end) {                       \
    size_t i = 0;                                                                       \
    for (auto it = begin; it != end; ++it) {                                            \
        if (**it % x == 0)                                                              \
            (*result)[i++] = *it;                                                       \
    }                                                                                   \
    *result_size = i;                                                                   \
}

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
    size_t num_elements = 12000000;
    size_t num_of_samples = 3;

    auto column = create_column(num_elements);

    cout << "type;data_set_size_mb;vector_size_num_int_val;duration_ms" << endl;

    for (auto &vector_size : /* ... */)
    {
        float duration = 0;
        /* sample this code multiple times according 'num_of_samples' */
            auto result = create_column(num_elements, false);
            /* measure execution time of this code block */
                auto mat = toolkit::materialize<int>(result);
                auto filter6 = toolkit::simple_filter<int>(&mat, vector_size, [](int *x) { return *x % 13 == 0; });
                auto filter5 = toolkit::simple_filter<int>(&filter6, vector_size, [](int *x) { return *x % 11 == 0; });
                auto filter4 = toolkit::simple_filter<int>(&filter5, vector_size, [](int *x) { return *x % 7 == 0; });
                auto filter3 = toolkit::simple_filter<int>(&filter4, vector_size, [](int *x) { return *x % 5 == 0; });
                auto filter2 = toolkit::simple_filter<int>(&filter3, vector_size, [](int *x) { return *x % 3 == 0; });
                auto filter1 = toolkit::simple_filter<int>(&filter2, vector_size, [](int *x) { return *x % 2 == 0; });
                auto read = toolkit::reader<int>(&filter1, column, column + num_elements, vector_size);
                read.start();
            /* end: measure execution time of this code block */
            duration += /* ... */
            delete_column(result);
        /* End: sample this code multiple times according 'num_of_samples' */
        cout << "simple;"
             << (float(num_elements * sizeof(int)) / 1024 / 1024) << ";"
             << vector_size << ";"
             << (duration / num_of_samples) << endl;
    }

    delete_column(column);

    return EXIT_SUCCESS;
}

