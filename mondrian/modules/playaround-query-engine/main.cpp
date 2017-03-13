
#include <vpipes.hpp>
#include <vpipes/functional.hpp>
#include <storage/column.hpp>
#include <utils/profiling.hpp>

using namespace mondrian;
using namespace mondrian::storage;

int main()
{
    size_t num_elements = 60000000;
    size_t vector_size = 4000;
    unsigned *data = (unsigned *) malloc (num_elements * sizeof(unsigned));
    for (size_t i = 0; i < num_elements; ++i) {
        data [i] = 2*i+1;
    }



    column<unsigned> col(data, data + num_elements);

    vpipes::functional::batched_materializes<unsigned>::func_t materialize = [&data] (unsigned *out_begin, unsigned *out_end,
                                                                       const size_t *begin, const size_t*end)
    {
        assert (out_end - out_begin >= end - begin);
        size_t distance = (end - begin);
        for (size_t i = 0; i != distance; ++i) {
            out_begin[i] = data[begin[i]];
        }
    };


    vpipes::toolkit::printer<unsigned> print(materialize);
    using pred = vpipes::functional::batched_predicates<unsigned>;

    for (size_t i = 0; i < 100; i++) {
        auto d = utils::profiling::measure<std::chrono::milliseconds>::execute([&col, &print, &vector_size]() {
            auto table_scan = col.table_scan(nullptr, pred::equal_to::straightforward_impl(25), vector_size);
            table_scan->start();
        });

        cout << d << "ms" << endl;
    }


    return EXIT_SUCCESS;
}