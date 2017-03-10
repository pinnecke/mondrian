
#include <vpipes.hpp>
#include <vpipes/functional.hpp>
#include <storage/column.hpp>

using namespace mondrian;
using namespace mondrian::storage;

int main()
{
    size_t num_elements = 100;
    size_t vector_size = 120;
    unsigned *data = (unsigned *) malloc (num_elements * sizeof(unsigned));
    for (size_t i = 0; i < num_elements; ++i) {
        data [i] = 2*i;
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

    using pred_t = vpipes::functional::batched_predicates<unsigned>;
    pred_t::func_t predicate = [] (pred_t::tupletid_t *result_buffer, size_t *result_size, const pred_t::value_t *begin,
                                   const pred_t::value_t *end)
    {
        for (pred_t::tupletid_t tid = 0; tid != (end - begin); ++tid) {
            if (begin[tid] % 2 == 0)
                result_buffer[(*result_size)++] = tid;
        }
    };

    vpipes::toolkit::printer<unsigned> print(materialize);
    auto table_scan = col.table_scan(&print, predicate, vector_size);
    table_scan->start();

    return EXIT_SUCCESS;
}