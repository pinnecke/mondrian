#include <iostream>

#include <storage/host/column_store.hpp>


int main() {

    pantheon::storage::host::column_store::base_column<uint16_t> c1("Ho", 2, 10, pantheon::utils::strings::to_string::uint16_to_string);
    uint16_t *values = (uint16_t *) malloc(100 * sizeof(uint16_t));
    for (size_t i = 0; i < 100; i++)
        values[i] = 2*i;

    c1.append(values, values + 100);
    c1.to_string(stdout);

    using tuplet_id_t = pantheon::storage::host::column_store::base_column<uint16_t>::tuplet_id_t;
    tuplet_id_t *tuplet_ids = (tuplet_id_t *) malloc(10 * sizeof(tuplet_id_t));
    for (int i = 9; i >= 0; --i)
        tuplet_ids[i] = 10 - i;

    c1.raw_print(stdout, tuplet_ids, tuplet_ids + 10);

    pantheon::storage::host::column_store::base_column<uint16_t>::mem_info info;
    c1.get_memory_info(&info);
    printf("\ncolumn_store_type_size=%zu, "
                   "number_of_data_pages=%zu, "
                   "total_approx_free_mask_size=%zu, "
                   "total_approx_null_mask_size=%zu, "
                   "total_flags_size=%zu, "
                   "total_link_size=%zu, "
                   "total_mutex_size=%zu, "
                   "total_payload_size=%zu\n",
            info.column_store_type_size,
            info.number_of_data_pages,
            info.total_approx_free_mask_size,
            info.total_approx_null_mask_size,
            info.total_flags_size,
            info.total_link_size,
            info.total_mutex_size,
            info.total_payload_size);


    return 0;
}