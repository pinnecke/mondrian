#include <iostream>
#include "framework/pipe_heads/reader.hpp"
#include "framework/pipes/sequential_filter.hpp"
#include "framework/pipe_tails/materialize.hpp"
#include "tasks.hpp"

using namespace std;
using namespace mondrian::query_engine::operators::sql;
using namespace mondrian::query_engine::operators::sources;
using namespace mondrian::query_engine::operators::sinks;

#define PREDICATE(x)                                                                    \
[] (int ***result, size_t *result_size, int **begin, int **end) {        \
           \
    size_t i = 0;                                                                       \
    for (auto it = begin; it != end; ++it) {                                            \
        if (**it % 2 == 0)                                                              \
            (*result)[i++] = *it;                                                               \
    }                                                                                   \
    *result_size = i;\
}

int main() {
    size_t num_elements = 20000000;
    size_t vector_size  = 72864;
    auto column = create_column(num_elements);

    auto d1 = measure<>::execute([&num_elements, &column] () {
        auto result = create_column(num_elements, false);
        int mod_values[] = { 2, 5, 7, 11, 13, 17, 19, 23};
        size_t idx_last = num_elements;
        for (int mod_value_idx = 0; mod_value_idx < 8; ++mod_value_idx) {
            size_t idx_current = 0;
            std::for_each(column, column + idx_last, [&result, &idx_last, &idx_current](int value) {
                if (value >= 0)
                    result[idx_current++] = value;
            });
            idx_last = idx_current;
        }
        delete_column(result);
    });

    cout << "STL filter w/ for_each:\t" << d1 << "ms" << endl;

    auto d2 = measure<>::execute([&num_elements, &column, &vector_size] () {
        auto result = create_column(num_elements, false);
        auto mat = materialize<int>(result);
        auto filter9 = sequential_filter<int>(&mat, vector_size, [] (int *x)     { return *x % 23 == 0; });
        auto filter8 = sequential_filter<int>(&filter9, vector_size, [] (int *x) { return *x % 19 == 0; });
        auto filter7 = sequential_filter<int>(&filter8, vector_size, [] (int *x) { return *x % 17 == 0; });
        auto filter6 = sequential_filter<int>(&filter7, vector_size, [] (int *x) { return *x % 13 == 0; });
        auto filter5 = sequential_filter<int>(&filter6, vector_size, [] (int *x) { return *x % 11 == 0; });
        auto filter4 = sequential_filter<int>(&filter5, vector_size, [] (int *x) { return *x % 7 == 0; });
        auto filter3 = sequential_filter<int>(&filter4, vector_size, [] (int *x) { return *x % 5 == 0; });
        auto filter2 = sequential_filter<int>(&filter3, vector_size, [] (int *x) { return *x % 3 == 0; });
        auto filter1 = sequential_filter<int>(&filter2, vector_size, [] (int *x) { return *x % 2 == 0; });
        auto read = reader<int>(&filter1, column, column + num_elements, vector_size);
        read.produce();
        delete_column(result);
    });

    cout << "Push-based vectorized:\t" << d2 << "ms" << endl;

    auto d3 = measure<>::execute([&num_elements, &column, &vector_size] () {
        auto result = create_column(num_elements, false);
        auto mat = materialize<int>(result);

     /*   auto filter9 = sequential_filter2<int>(&mat, vector_size, PREDICATE(23));
        auto filter8 = sequential_filter2<int>(&filter9, vector_size, PREDICATE(19));
        auto filter7 = sequential_filter2<int>(&filter8, vector_size, PREDICATE(17));
        auto filter6 = sequential_filter2<int>(&filter7, vector_size, PREDICATE(13));
        auto filter5 = sequential_filter2<int>(&filter6, vector_size, PREDICATE(11));
        auto filter4 = sequential_filter2<int>(&filter5, vector_size, PREDICATE(7));
        auto filter3 = sequential_filter2<int>(&filter4, vector_size, PREDICATE(5));
        auto filter2 = sequential_filter2<int>(&filter3, vector_size, PREDICATE(3));
        auto filter1 = sequential_filter2<int>(&filter2, vector_size, PREDICATE(2));*/
        auto filter1 = sequential_filter2<int>(&mat, vector_size, PREDICATE(2));
        auto read = reader<int>(&filter1, column, column + num_elements, vector_size);
        read.produce();
        delete_column(result);
    });

    cout << "Push-based vectorized (vectorized pred):\t" << d3 << "ms" << endl;

    delete_column(column);

    return EXIT_SUCCESS;
}

