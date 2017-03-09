
#include <vpipes.hpp>
#include <storage/column.hpp>

using namespace mondrian;
using namespace mondrian::storage;

int main()
{
    size_t num_elements = 2e6;
    size_t vector_size = 4096;
    unsigned *data = (unsigned *) malloc (num_elements * sizeof(unsigned));
    for (size_t i = 0; i < num_elements; ++i) {
        data [i] = i;
    }

    vpipes::toolkit::printer<unsigned> print;

    column<unsigned> col(data, data + num_elements);

    auto table_scan = col.table_scan(&print, vector_size);
    table_scan->start();

    return EXIT_SUCCESS;
}