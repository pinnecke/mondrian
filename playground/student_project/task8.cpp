#include <iostream>

#include <vpipes.hpp>
#include <thread>

#include "tasks.hpp"

using namespace std;
using namespace mondrian::vpipes;

template<class Type, class ForwardIt>
class intra_parallel_simple_filter;

template<class T, class F = T*>
struct thread_function
{
    using input_iterator_t = F;


   // std::function<bool(input_iterator_t)> pred;
    intra_parallel_simple_filter<T, F> *context;
    size_t num_of_threads;
    input_iterator_t *begin, *end;
    input_iterator_t **result_buffer;
    size_t result_buffer_size;
    size_t thread_id;

    thread_function(std::function<bool(input_iterator_t)> predicate,
                    intra_parallel_simple_filter<T, F> *context,
                    size_t num_of_threads, size_t thread_id, input_iterator_t *begin, input_iterator_t *end):
            /*pred(predicate),*/ context(context), num_of_threads(num_of_threads), begin(begin), end(end),
            thread_id(thread_id), result_buffer(nullptr), result_buffer_size(0) { }

    void invoke() {
        assert (context != nullptr);
        size_t vector_size = end - begin;
        result_buffer = (input_iterator_t **) malloc (vector_size * sizeof(input_iterator_t *));
        result_buffer_size = 0;

        size_t chunk_size = size_t(std::ceil(vector_size / (float) num_of_threads));;
        auto chunk_begin = chunk_size * thread_id;
        auto chunk_end = std::min(chunk_begin + chunk_size, vector_size);

        for (auto it = begin + chunk_begin; it != begin + chunk_end; ++it) {
           // if (pred(*it)) {
            if (**it % (thread_id+1) == 0) {
                result_buffer[result_buffer_size++] = it;
            }
            // }
        }
    }
};

template<class Type, class ForwardIt = Type*>
class intra_parallel_simple_filter : public pipe<Type, Type, ForwardIt, ForwardIt>
{
    using super = pipe<Type, Type, ForwardIt, ForwardIt>;

    template<class T, class F> friend
    struct thread_function;

public:
    using typename super::input_t;
    using typename super::input_iterator_t;
    using typename super::consumer_t;



    std::function<bool(input_iterator_t)> predicate;
    size_t num_of_threads;
public:

    intra_parallel_simple_filter(consumer_t *consumer, unsigned vector_size,
                  function<bool(input_iterator_t)> predicate, size_t num_of_threads = 8) :
            super(consumer, vector_size), predicate(predicate), num_of_threads(num_of_threads)
    { }

    virtual void on_consume(input_iterator_t *begin, input_iterator_t *end) override
    {
        thread_function<input_t, input_iterator_t>* thread_functions = (thread_function<input_t, input_iterator_t>*) malloc(num_of_threads * sizeof(thread_function<input_t, input_iterator_t>));
        std::vector<std::thread> threads(num_of_threads);

        for (unsigned thread_id = 0; thread_id < num_of_threads; ++thread_id) {
            thread_functions[thread_id] = thread_function<input_t, input_iterator_t>(predicate, this, num_of_threads,
                                                                                         thread_id,
                                                                                         begin, end);
            auto tf = &thread_functions[thread_id];

            threads[thread_id] = std::thread([this, &thread_id, &tf, &begin, &end] ()
            {

                tf->invoke();
            });
        }

        for (unsigned thread_id = 0; thread_id < num_of_threads; ++thread_id) {
            threads[thread_id].join();
            for (size_t i = 0; i < thread_functions[thread_id].result_buffer_size; ++i) {
                super::produce(thread_functions[thread_id].result_buffer[i]);
            }
        }

        free (thread_functions);
    }
};


int main() {
    size_t num_elements = 120000000;
    size_t vector_size  = 32768;
    size_t num_of_samples = 3;
    auto column = create_column(num_elements, false, true);

    cout << "type;data_set_size_mb;vector_size_num_int_val;duration_ms" << endl;

    for (auto &vector_size : {27310, 32768, 40965, 61447, 92170, 138255, 207382, 311073, 466609, 699913, 1049869, 1574803, 2362204})
    {
        float duration = 0;
        /*for (auto sample = 0; sample < num_of_samples; ++sample) {
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
*/


        duration = 0;
        for (auto sample = 0; sample < num_of_samples; ++sample) {
            auto result = create_column(num_elements, false);
            duration += measure<>::execute([&num_elements, &column, &vector_size, &result]() {
                auto mat = toolkit::materialize<int>(result);
                auto filter6 = intra_parallel_simple_filter<int>(&mat, vector_size, [](int *x) { return *x % 13 == 0; });
                auto filter5 = intra_parallel_simple_filter<int>(&filter6, vector_size, [](int *x) { return *x % 11 == 0; });
                auto filter4 = intra_parallel_simple_filter<int>(&filter5, vector_size, [](int *x) { return *x % 7 == 0; });
                auto filter3 = intra_parallel_simple_filter<int>(&filter4, vector_size, [](int *x) { return *x % 5 == 0; });
                auto filter2 = intra_parallel_simple_filter<int>(&filter3, vector_size, [](int *x) { return *x % 3 == 0; });
                auto filter1 = intra_parallel_simple_filter<int>(&filter2, vector_size, [](int *x) { return *x % 2 == 0; });
                auto read = toolkit::reader<int>(&filter1, column, column + num_elements, vector_size);
                read.start();
            });
            delete_column(result);
        }
        cout << "intra_parallel;"
             << (float(num_elements * sizeof(int)) / 1024 / 1024) << ";"
             << vector_size << ";"
             << (duration / num_of_samples) << endl;


    }

    delete_column(column);


    return EXIT_SUCCESS;
}

