#include <iostream>
#include <numeric>
#include <mondrian>

#include "generic_filter.hpp"
#include "counter.hpp"

using namespace mondrian;
using namespace mondrian::utils::profiling;

using namespace mondrian::query_engine::operators;

template<class InputType, class InputPointerType = InputType*>
class sample_filter_is_even : public push_operator<InputType, InputPointerType> {
    using super = push_operator<InputType, InputPointerType>;
public:
    using typename super::input_t;
    using typename super::input_pointer_t;

    sample_filter_is_even(super *consumer, unsigned vector_size) :
            super(consumer, vector_size) { }

    virtual void on_consume(const input_pointer_t *begin, const input_pointer_t *end) override {
        for (auto it = begin; it != end; ++it) {
            if (super::lookup(it) % 2 == 0)
                super::forward(it);
        }
    }

};

int main() {

    std::cout << "Program is starting..." << std::endl;

    size_t num_of_values = 1000000000;
    unsigned *begin, *end;

    std::cout << "Allocate memory for column...";
    auto exec_malloc = measure<std::chrono::milliseconds>::execute([&num_of_values, &begin, &end] () {
        begin = (unsigned int *) malloc (num_of_values * sizeof(unsigned));
        end = begin + num_of_values;
    });
    std::cout << "Done (" << exec_malloc << "ms)" << std::endl;

    std::cout << "Fill column with data...";
    auto exec_fill = measure<std::chrono::milliseconds>::execute([&begin, &end] () {
        std::iota (begin, end, 0);
    });
    std::cout << "Done (" << exec_fill << "ms)" << std::endl;

    std::cout << "Invoke query..." << std::endl;
    auto exec_duration = measure<std::chrono::milliseconds>::execute([&begin, &end] () {
              using namespace query_engine::operators::sources;
              using namespace query_engine::operators::sql;
              using namespace query_engine::operators::sinks;

              auto print   = printer<unsigned>();
              auto sum     = sequential_sum<unsigned>(&print, 1000000);
              auto count   = counter<unsigned>(&sum, 1000000);
              auto filter2 = sample_filter_is_even<unsigned>(&count, 1000000);
              auto filter1 = generic_filter<unsigned>(&filter2, 1000000,[] (const unsigned *x) { return (*x) < 100; });
              auto read    = reader<unsigned>(&filter1, begin, end, 1000000);

              read.produce();
    });
    std::cout << "Done (" << exec_duration << "ms)" << std::endl;

    std::cout << "Cleanup...";
    auto exec_free = measure<std::chrono::milliseconds>::execute([&begin] () {
        free (begin);
    });
    std::cout << "Done (" << exec_free << "ms)" << std::endl;

    std::cout << "Exit" << std::endl;
    return EXIT_SUCCESS;
}

