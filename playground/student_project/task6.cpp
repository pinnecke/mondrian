#include <iostream>
#include <vpipes.hpp>
#include "tasks.hpp"

using namespace std;
using namespace mondrian::vpipes;

int main() {
    size_t num_elements = 2000000;
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

    auto d2 = measure<>::execute([&num_elements, &column, &vector_size] () {
        auto result = create_column(num_elements, false);
        auto mat = toolkit::materialize<int>(result);
        auto filter9 = toolkit::simple_filter<int>(&mat, vector_size, [] (int *x)     { return *x % 23 == 0; });
        auto filter8 = toolkit::simple_filter<int>(&filter9, vector_size, [] (int *x) { return *x % 19 == 0; });
        auto filter7 = toolkit::simple_filter<int>(&filter8, vector_size, [] (int *x) { return *x % 17 == 0; });
        auto filter6 = toolkit::simple_filter<int>(&filter7, vector_size, [] (int *x) { return *x % 13 == 0; });
        auto filter5 = toolkit::simple_filter<int>(&filter6, vector_size, [] (int *x) { return *x % 11 == 0; });
        auto filter4 = toolkit::simple_filter<int>(&filter5, vector_size, [] (int *x) { return *x % 7 == 0; });
        auto filter3 = toolkit::simple_filter<int>(&filter4, vector_size, [] (int *x) { return *x % 5 == 0; });
        auto filter2 = toolkit::simple_filter<int>(&filter3, vector_size, [] (int *x) { return *x % 3 == 0; });
        auto filter1 = toolkit::simple_filter<int>(&filter2, vector_size, [] (int *x) { return *x % 2 == 0; });
        auto read = toolkit::reader<int>(&filter1, column, column + num_elements, vector_size);
        read.start();
        delete_column(result);
    });

    cout << "STL filter w/ for_each:\t" << d1 << "ms" << endl;
    cout << "Push-based vectorized:\t" << d2 << "ms" << endl;

    delete_column(column);

    return EXIT_SUCCESS;
}

