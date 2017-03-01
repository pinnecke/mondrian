#include <iostream>
#include "framework/pipe_heads/reader.hpp"
#include "framework/pipes/filters.hpp"
#include "framework/pipe_tails/materialize.hpp"
#include "tasks.hpp"
#include "framework/pipe_tails/printer.hpp"

using namespace std;
using namespace mondrian::query_engine::operators::sql;
using namespace mondrian::query_engine::operators::sources;
using namespace mondrian::query_engine::operators::sinks;

#define PREDICATE(x)                                                                    \
[] (int ***result, size_t *result_size, int **begin, int **end) {        \
           \
    size_t i = 0;                                                                       \
    for (auto it = begin; it != end; ++it) {                                            \
        if (**it > x)                                                              \
            (*result)[i++] = *it;                                                               \
    }                                                                                   \
    *result_size = i;\
}

int main() {
    size_t num_elements = 100000000;
    size_t vector_size  = 72864;
    auto column = create_column(num_elements, false, true);
    size_t idx_last = num_elements;

    auto d1 = measure<>::execute([&num_elements, &column, &idx_last] () {
        auto result1 = std::vector<int>();
        auto result2 = std::vector<int>();
        result1.reserve(num_elements);
        result2.reserve(num_elements);

        std::copy_if(column, column + num_elements, std::back_inserter(result1), [](const int& i){ return i > 10;});

        result2.clear();
        result2.reserve(num_elements);
        std::copy_if(result1.begin(), result1.end(), std::back_inserter(result2), [](const int& i){ return i > 100;});
        result1.clear();
        result1.reserve(num_elements);
        std::copy_if(result2.begin(), result2.end(), std::back_inserter(result1), [](const int& i){ return i > 1000;});
        result2.clear();
        result2.reserve(num_elements);
        std::copy_if(result1.begin(), result1.end(), std::back_inserter(result2), [](const int& i){ return i > 10000;});
        result1.clear();
        result1.reserve(num_elements);
        std::copy_if(result2.begin(), result2.end(), std::back_inserter(result1), [](const int& i){ return i > 100000;});
        result2.clear();
        result2.reserve(num_elements);
        std::copy_if(result1.begin(), result1.end(), std::back_inserter(result2), [](const int& i){ return i > 1000000;});
        result1.clear();
        result1.reserve(num_elements);
        std::copy_if(result2.begin(), result2.end(), std::back_inserter(result1), [](const int& i){ return i > 2000000;});
        result2.clear();
        result2.reserve(num_elements);
        std::copy_if(result1.begin(), result1.end(), std::back_inserter(result2), [](const int& i){ return i > 99999990;});

       // cout << ">> " << (result2.size() > 0 ? result2[0] : 42) << " size: " << result2.size() << endl;
    });


    cout << "STL filter:\t\t" << d1 << "ms" << endl;

    auto d2 = measure<>::execute([&num_elements, &column, &vector_size] () {
        auto result = create_column(num_elements, false, false);
        auto mat = materialize<int>(result);
        auto print = printer<int>();


//        auto filter9 = sequential_filter<int>(&print, vector_size, [] (int *x)     { return *x % 23 == 0; });
        auto filter8 = simple_filter<int>(&mat, vector_size, [] (int *x)   { return *x > 99999990; });
        auto filter7 = simple_filter<int>(&filter8, vector_size, [] (int *x) { return *x > 2000000; });
        auto filter6 = simple_filter<int>(&filter7, vector_size, [] (int *x) { return *x > 1000000; });
        auto filter5 = simple_filter<int>(&filter6, vector_size, [] (int *x) { return *x > 100000; });
        auto filter4 = simple_filter<int>(&filter5, vector_size, [] (int *x) { return *x > 10000; });
        auto filter3 = simple_filter<int>(&filter4, vector_size, [] (int *x) { return *x > 1000; });
        auto filter2 = simple_filter<int>(&filter3, vector_size, [] (int *x) { return *x > 100; });
        auto filter1 = simple_filter<int>(&filter2, vector_size, [] (int *x) { return *x > 10; });
        auto read = reader<int>(&filter1, column, column + num_elements, vector_size);
        read.start();
        delete_column(result);
    });

    cout << "Push-based vectorized:\t" << d2 << "ms" << endl;

    auto d3 = measure<>::execute([&num_elements, &column, &vector_size] () {
        auto result = create_column(num_elements, false, false);
        auto mat = materialize<int>(result);

       // auto print = printer<int>();
        auto filter8 = batched_pred_filter<int>(&mat, vector_size, PREDICATE(99999990));
        auto filter7 = batched_pred_filter<int>(&filter8, vector_size, PREDICATE(2000000));
        auto filter6 = batched_pred_filter<int>(&filter7, vector_size, PREDICATE(1000000));
        auto filter5 = batched_pred_filter<int>(&filter6, vector_size, PREDICATE(100000));
        auto filter4 = batched_pred_filter<int>(&filter5, vector_size, PREDICATE(10000));
        auto filter3 = batched_pred_filter<int>(&filter4, vector_size, PREDICATE(1000));
        auto filter2 = batched_pred_filter<int>(&filter3, vector_size, PREDICATE(100));
        auto filter1 = batched_pred_filter<int>(&filter2, vector_size, PREDICATE(10));
        auto read = reader<int>(&filter1, column, column + num_elements, vector_size);
        read.start();
        delete_column(result);
    });

    cout << "Push-based vectorized (vectorized pred):\t" << d3 << "ms" << endl;

    delete_column(column);

    return EXIT_SUCCESS;
}

