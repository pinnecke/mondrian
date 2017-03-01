#pragma once

#include <iostream>
#include <limits>
#include <random>
#include <cassert>
#include "framework/pipe.hpp"

#include "profiling.hpp"

using namespace std;
using namespace mondrian::query_engine::operators;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Task 1
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

default_random_engine generator;

int random_number(int lower_bound = numeric_limits<int>::min(), int upper_bound = numeric_limits<int>::max())
{
    static uniform_int_distribution<int> distribution(lower_bound, upper_bound);
    return distribution(generator);
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Task 2
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int *create_column(unsigned long num_of_elements, bool fill_with_random = true)
{
    assert (num_of_elements > 0);
    auto result = (int *) malloc (num_of_elements * sizeof(int));
    if (fill_with_random) {
        for (auto i = 0; i < num_of_elements; ++i)
            result[i] = random_number();
    }
    return result;
}

void delete_column(int *column)
{
    assert (column != nullptr);
    free (column);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Task 7
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
struct XXX
{
    T **begin;
    T **end;

    XXX(): begin(nullptr), end(nullptr) { }
    XXX(T **begin, T **end): begin(begin), end(end) { }
};

template<class Type, class ForwardIt = Type*>
class sequential_filter2 : public pipe<Type, Type, ForwardIt, ForwardIt>
{
    using super = pipe<Type, Type, ForwardIt, ForwardIt>;
public:
    using typename super::input_t;
    using typename super::input_iterator_t;
    using typename super::consumer_t;
    using iterator_t = mondrian::query_engine::operators::iterator<input_iterator_t *>;

private:
    input_iterator_t *result_buffer;
    size_t result_buffer_size;

    function<void(input_iterator_t **result, size_t *result_size, input_iterator_t *begin, input_iterator_t *end)> predicate;
public:

    sequential_filter2(consumer_t *consumer, unsigned vector_size,
            function<void(input_iterator_t **result, size_t *result_size, input_iterator_t *begin, input_iterator_t *end)> predicate) :
    super(consumer, vector_size), predicate(predicate)
    {
        result_buffer_size = vector_size; // Note here: that the operator is unaware of the vectorize of the input. This assignment is just a best guess and must be corrected afterwards if it was wrong
        result_buffer = (input_iterator_t *) malloc(result_buffer_size * sizeof(input_iterator_t));
        assert (result_buffer != nullptr);
    }

    virtual void on_consume(input_iterator_t *begin, input_iterator_t *end) override
    {
        auto vector_size = (end - begin);
        if (vector_size > result_buffer_size) {
            vector_size = result_buffer_size;
            result_buffer = (input_iterator_t *) realloc(result_buffer, vector_size * sizeof(input_iterator_t));
            assert (result_buffer != nullptr);
        }

        size_t result_size;
        predicate(&result_buffer, &result_size, begin, end);

        for (auto it = result_buffer; it != result_buffer + result_size; ++it) {
                super::forward(it);
        }
    }

    virtual void on_cleanup() override
    {
        free (result_buffer);
    }
};

