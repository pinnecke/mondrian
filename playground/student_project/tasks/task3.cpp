#include <iostream>
#include <vpipes.hpp>

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

int random_number(int lower_bound /* ... ? */, int upper_bound /* ... ? */)
{
    // TODO: add code here
};

int *create_column(unsigned long num_of_elements, bool fill_with_random = true, bool fill = true)
{
    // TODO: add code here
}

void delete_column(int *column)
{
    // TODO: add code here
}

int main() {
    size_t num_elements = 120000000;
    size_t num_of_samples = 3;

    auto column = create_column(num_elements);

    cout << "type;data_set_size_mb;vector_size_num_int_val;duration_ms" << endl;

    /* Set vector_size from the range { 1, 2, 3, ... } */
            auto result = create_column(num_elements, false);
            /* Sample the execution times of this multiples times. */
                auto mat = toolkit::materialize<int>(result);
                auto filter6 = toolkit::simple_filter<int>(&mat, vector_size, [](int *x) { return *x % 13 == 0; });
                auto filter5 = toolkit::simple_filter<int>(&filter6, vector_size, [](int *x) { return *x % 11 == 0; });
                auto filter4 = toolkit::simple_filter<int>(&filter5, vector_size, [](int *x) { return *x % 7 == 0; });
                auto filter3 = toolkit::simple_filter<int>(&filter4, vector_size, [](int *x) { return *x % 5 == 0; });
                auto filter2 = toolkit::simple_filter<int>(&filter3, vector_size, [](int *x) { return *x % 3 == 0; });
                auto filter1 = toolkit::simple_filter<int>(&filter2, vector_size, [](int *x) { return *x % 2 == 0; });
                auto read = toolkit::reader<int>(&filter1, column, column + num_elements, vector_size);
                read.start();
            /* End: Sample the execution times of this multiples times. */
            delete_column(result);

            auto result = create_column(num_elements, false);
            /* Sample the execution times of this multiples times. */
                auto mat = toolkit::materialize<int>(result);
                auto filter6 = toolkit::batched_pred_filter<int>(&mat, vector_size, PREDICATE(13));
                auto filter5 = toolkit::batched_pred_filter<int>(&filter6, vector_size, PREDICATE(11));
                auto filter4 = toolkit::batched_pred_filter<int>(&filter5, vector_size, PREDICATE(7));
                auto filter3 = toolkit::batched_pred_filter<int>(&filter4, vector_size, PREDICATE(5));
                auto filter2 = toolkit::batched_pred_filter<int>(&filter3, vector_size, PREDICATE(3));
                auto filter1 = toolkit::batched_pred_filter<int>(&filter2, vector_size, PREDICATE(2));
                auto read = toolkit::reader<int>(&filter1, column, column + num_elements, vector_size);
                read.start();
            /* End: Sample the execution times of this multiples times. */
            delete_column(result);
    /* End: Set vector_size from the range { 1, 2, 3, ... } */

    delete_column(column);

    return EXIT_SUCCESS;
}

