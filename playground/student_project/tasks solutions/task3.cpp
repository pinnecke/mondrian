#include <iostream>
#include <vpipes.hpp>
#include "tasks.hpp"

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

int main() {
    size_t num_elements = 120000000;
    size_t num_of_samples = 3;

    auto column = create_column(num_elements);

    cout << "type;data_set_size_mb;vector_size_num_int_val;duration_ms" << endl;

    for (auto &vector_size : {1, 2, 3, 4, 6, 9, 13, 19, 28, 42, 63, 94, 141, 211, 316, 474, 711, 1066, 1599, 2398, 3597, 5395, 8092, 12138, 18207, 27310, 40965, 61447, 92170, 138255, 207382, 311073, 466609, 699913, 1049869, 1574803, 2362204})
    {
        float duration = 0;
        for (auto sample = 0; sample < num_of_samples; ++sample) {
            auto result = create_column(num_elements, false);
            duration += measure<>::execute([&num_elements, &column, &vector_size, &result]() {
                auto mat = toolkit::materialize<int>(result);
                auto filter6 = toolkit::simple_filter<int>(&mat, vector_size, [](int *x) { return *x % 13 == 0; });
                auto filter5 = toolkit::simple_filter<int>(&filter6, vector_size, [](int *x) { return *x % 11 == 0; });
                auto filter4 = toolkit::simple_filter<int>(&filter5, vector_size, [](int *x) { return *x % 7 == 0; });
                auto filter3 = toolkit::simple_filter<int>(&filter4, vector_size, [](int *x) { return *x % 5 == 0; });
                auto filter2 = toolkit::simple_filter<int>(&filter3, vector_size, [](int *x) { return *x % 3 == 0; });
                auto filter1 = toolkit::simple_filter<int>(&filter2, vector_size, [](int *x) { return *x % 2 == 0; });
                auto read = toolkit::reader<int>(&filter1, column, column + num_elements, vector_size);
                read.start();
            });
            delete_column(result);
        }
        cout << "simple;"
             << (float(num_elements * sizeof(int)) / 1024 / 1024) << ";"
             << vector_size << ";"
             << (duration / num_of_samples) << endl;

        duration = 0;
        for (auto sample = 0; sample < num_of_samples; ++sample) {
            auto result = create_column(num_elements, false);
            duration += measure<>::execute([&num_elements, &column, &vector_size, &result]() {
                auto mat = toolkit::materialize<int>(result);
                auto filter6 = toolkit::batched_pred_filter<int>(&mat, vector_size, PREDICATE(13));
                auto filter5 = toolkit::batched_pred_filter<int>(&filter6, vector_size, PREDICATE(11));
                auto filter4 = toolkit::batched_pred_filter<int>(&filter5, vector_size, PREDICATE(7));
                auto filter3 = toolkit::batched_pred_filter<int>(&filter4, vector_size, PREDICATE(5));
                auto filter2 = toolkit::batched_pred_filter<int>(&filter3, vector_size, PREDICATE(3));
                auto filter1 = toolkit::batched_pred_filter<int>(&filter2, vector_size, PREDICATE(2));
                auto read = toolkit::reader<int>(&filter1, column, column + num_elements, vector_size);
                read.start();
            });
            delete_column(result);
        }
        cout << "batched;"
             << (float(num_elements * sizeof(int)) / 1024 / 1024) << ";"
             << vector_size << ";"
             << (duration / num_of_samples) << endl;
    }

    delete_column(column);

    return EXIT_SUCCESS;
}

