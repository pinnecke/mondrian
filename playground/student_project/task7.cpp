#include <iostream>

#include <vpipes.hpp>

#include "tasks.hpp"

using namespace std;
using namespace mondrian::vpipes;

#define PREDICATE(x)                                                                    \
[] (int ***result, size_t *result_size, int **begin, int **end) {                       \
                                                                                        \
    size_t i = 0;                                                                       \
    for (auto it = begin; it != end; ++it) {                                            \
        if (**it > x)                                                                   \
            (*result)[i++] = *it;                                                       \
    }                                                                                   \
    *result_size = i;\
}

int main() {
    size_t num_elements = 16000000;
    size_t vector_size  = 65536;
    auto column = create_column(num_elements, false, true);
    size_t idx_last = num_elements;

    auto d2 = measure<>::execute([&num_elements, &column, &vector_size] () {
        auto result = create_column(num_elements, false, false);
        auto mat = toolkit::materialize<int>(result);
        auto print = toolkit::printer<int>();

        auto filter8 = toolkit::simple_filter<int>(&print, vector_size, [] (int *x)   { return *x > 15999998; });
        auto filter7 = toolkit::simple_filter<int>(&filter8, vector_size, [] (int *x) { return *x > 1600000; });
        auto filter6 = toolkit::simple_filter<int>(&filter7, vector_size, [] (int *x) { return *x > 160000; });
        auto filter5 = toolkit::simple_filter<int>(&filter6, vector_size, [] (int *x) { return *x > 16000; });
        auto filter4 = toolkit::simple_filter<int>(&filter5, vector_size, [] (int *x) { return *x > 1600; });
        auto filter3 = toolkit::simple_filter<int>(&filter4, vector_size, [] (int *x) { return *x > 160; });
        auto filter2 = toolkit::simple_filter<int>(&filter3, vector_size, [] (int *x) { return *x > 16; });
        auto filter1 = toolkit::simple_filter<int>(&filter2, vector_size, [] (int *x) { return *x > 1; });
        auto read = toolkit::reader<int>(&filter1, column, column + num_elements, vector_size);
        read.start();
        delete_column(result);
    });

    cout << "Push-based vectorized:\t" << d2 << "ms" << endl;


//    for (int i = 1; i < 100; i += 1) {

        auto d3 = measure<>::execute([&num_elements, &column, &vector_size]() {
            auto result = create_column(num_elements, false, false);
            auto mat = toolkit::materialize<int>(result);
            auto print = toolkit::printer<int>();

            // auto print = printer<int>();
            auto filter8 = toolkit::batched_pred_filter<int>(&print, vector_size, PREDICATE(15999998));
            auto filter7 = toolkit::batched_pred_filter<int>(&filter8, vector_size, PREDICATE(1600000));
            auto filter6 = toolkit::batched_pred_filter<int>(&filter7, vector_size, PREDICATE(160000));
            auto filter5 = toolkit::batched_pred_filter<int>(&filter6, vector_size, PREDICATE(16000));
            auto filter4 = toolkit::batched_pred_filter<int>(&filter5, vector_size, PREDICATE(1600));
            auto filter3 = toolkit::batched_pred_filter<int>(&filter4, vector_size, PREDICATE(160));
            auto filter2 = toolkit::batched_pred_filter<int>(&filter3, vector_size, PREDICATE(16));
            auto filter1 = toolkit::batched_pred_filter<int>(&filter2, vector_size, PREDICATE(1));
            auto read = toolkit::reader<int>(&filter1, column, column + num_elements, vector_size);
            read.start();
            delete_column(result);
        });

        cout << d3<< endl;
    //   }

    delete_column(column);

    return EXIT_SUCCESS;
}

