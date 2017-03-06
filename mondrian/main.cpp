extern "C" {
#include <modules/manifest.h>
}

begin_modules()
begin_module_context("<command> <args>",
"You must specify a command to run. See the command list below.",
"These are common MondrianDB commands used in various situations:");

begin_group("Information about the system and the systems environment")
install_module(version, "version", "Displays version information about the current build");
   install_module(show, "show", "Lists information about the license.");
       install_man_page("c", "mnd_c");
      install_man_page("w", "mnd_w");
end_group()

//begin_group("Experimental, and research related features")
//    install_module(exp, "exp", "Experimental features which are work-in-progress");
//end_group()



//install_man_page("exp cnshell", "rcn_cnshell");

/*begin_group("Displays configuration settings, monitor and more.")
    install_module(show);
end_group()*/



end_module_context();
end_modules()

/*#include <iostream>

#include <storage/host/column_store.hpp>


int main() {

    pantheon::storage::host::column_store::base_column<uint16_t> c1("Ho", 2, 10, pantheon::utils::strings::to_string::uint16_to_string);
    uint16_t *values = (uint16_t *) malloc(100 * sizeof(uint16_t));
    for (size_t i = 0; i < 100; i++)
        values[i] = i;

    c1.append(values, values + 100);
    c1.to_string(stdout);

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
}*/