#include <iostream>
#include <numeric>

#include <mondrian>

using namespace mondrian;
using namespace mondrian::query_engine::operators;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Implementation task 1
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<class ValueType>
class my_filter_less_than_five : public push_operator<ValueType> {
    using super = push_operator<ValueType>;
public:
    my_filter_less_than_five(super *consumer, unsigned vector_size) : super(consumer, vector_size) {}

    virtual void on_consume(const ValueType **begin, const ValueType **end) override {

        // TODO: Implementation
        // Implement an (sequential) iteration from 'begin' to 'end' and extract elements that are less than five.
        // Tipp: - For each visited element 'e' invoke super::lookup(e) to get the value behind 'e'.
        //       - To pass a satisfying element to subsequent operators, use super::yield(e) for an element e

    }
};

int main() {

    size_t num_of_values = 10000000;
    unsigned *values = (unsigned int *) malloc (num_of_values * sizeof(unsigned));
    unsigned *begin = values, *end = values + num_of_values;
    std::iota (begin, end, 0);



    auto print = query_engine::operators::sinks::printer<unsigned>();

    auto scan = my_filter_less_than_five<unsigned>(&print, 10);

    auto read = query_engine::operators::sources::reader<unsigned>(&scan, begin, end, 10);

    read.produce();

    free (values);
    return EXIT_SUCCESS;
}