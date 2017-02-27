#include <iostream>
#include <numeric>
#include <mondrian>

#include "generic_filter.hpp"
#include "counter.hpp"

using namespace mondrian;
using namespace mondrian::utils::profiling;
using namespace query_engine::operators::sources;
using namespace query_engine::operators::sql;
using namespace query_engine::operators::sinks;

using namespace mondrian::query_engine::operators;

template<class InputType, class OutputType, class InputPointerType = InputType *, class OutputPointerType = OutputType *>
class sample_filter_is_even : public pipe<InputType, OutputType, InputPointerType,
        OutputPointerType> {
    using super = pipe<InputType, OutputType, InputPointerType, OutputPointerType>;
public:
    using typename super::input_t;
    using typename super::input_iterator_t;
    using typename super::consumer_t;

    sample_filter_is_even(consumer_t *consumer, unsigned vector_size) :
            super(consumer, vector_size) { }

    virtual void on_consume(const input_iterator_t *begin, const input_iterator_t *end) override {
        for (auto it = begin; it != end; ++it) {
            if (super::lookup(it) % 2 == 0)
                super::forward(it);
        }
    }

};

int main() {

    std::cout << "Program is starting..." << std::endl;

    size_t num_of_values = 1000000000;
    size_t vector_size = 262144;
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

    std::cout << "Invoke query #1..." << std::endl;
    auto exec_duration = measure<std::chrono::milliseconds>::execute([&begin, &end, &vector_size] () {


        auto x = sql::sequential_sum<unsigned, unsigned>(nullptr, 10);
        auto y = sql::sequential_count<unsigned, unsigned>(nullptr, 10);
        auto z = sql::sequential_filter<unsigned>(nullptr, 10, [] (const unsigned *x) { return (*x) < 100; });

        auto print   = printer<unsigned>();
        auto sum     = sequential_sum<unsigned, unsigned>(&print, vector_size);
        auto count   = counter<unsigned, unsigned>(&sum, vector_size);
        auto filter2 = sample_filter_is_even<unsigned, unsigned>(&count, vector_size);
        auto filter1 = generic_filter<unsigned, unsigned>(&filter2, vector_size,[] (const unsigned *x) { return (*x) < 100; });
        auto read    = reader<unsigned>(&filter1, begin, end, vector_size);

        read.produce();
    });
    std::cout << "Done (" << exec_duration << "ms)" << std::endl;

    std::cout << "Invoke query #2..." << std::endl;
    auto exec_duration2 = measure<std::chrono::milliseconds>::execute([&begin, &end, &vector_size] () {

        auto print   = printer<unsigned>();
        auto join    = test_bi_pipe<unsigned, unsigned, unsigned>(&print, vector_size);

        auto read1    = reader<unsigned>(join.get_left_port(), begin, end, vector_size);
        auto read2    = reader<unsigned>(join.get_right_port(), begin, end, vector_size);

        read1.produce();
        read2.produce();
    });
    std::cout << "Done (" << exec_duration2 << "ms)" << std::endl;

    std::cout << "Cleanup...";
    auto exec_free = measure<std::chrono::milliseconds>::execute([&begin] () {
        free (begin);
    });
    std::cout << "Done (" << exec_free << "ms)" << std::endl;

    std::cout << "Exit" << std::endl;
    return EXIT_SUCCESS;
}

