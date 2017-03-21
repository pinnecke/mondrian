
#include <vpipes.hpp>
#include <vpipes/functional.hpp>
#include <storage/column.hpp>
#include <utils/profiling.hpp>
#include <fstream>
#include <vector>
#include <sys/stat.h>

using namespace mondrian;
using namespace mondrian::storage;

bool file_exists(const char *fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

void get_column_file_path(std::string *path, const std::string &column_label)
{
    if (!file_exists(path->c_str())) {
        cout << "Enter absolute path to binary " << column_label << " (uint32) file:" << endl;
        getline(cin, *path);
    } else {
        std::string choice;
        cout << "Use default path to " << column_label << " binary file? \nDefault path is `"<< *path <<"`.\n(yes/no)" << endl;
        getline(cin, choice);
        std::transform(choice.begin(), choice.end(), choice.begin(), ::tolower);
        if (choice.compare("yes") != 0 && choice.compare("y") != 0) {
            cout << "Enter path: ";
            getline(cin, *path);
        }
    }

    if (!file_exists(path->c_str())) {
        cerr << "File does not exists: '" << *path << "'.\nSTOP.\n";
        cerr.flush();
        exit (EXIT_FAILURE);
    }
}

long get_file_size(std::string file_name)
{
    struct stat stat_buf;
    int result = stat(file_name.c_str(), &stat_buf);
    return result == 0 ? stat_buf.st_size : -1;
}

template <typename Type>
std::vector<Type> read_from_file(std::string file_name)
{
    struct stat info;
    if (stat(file_name.c_str(), &info) != 0) {
        cerr << "Unable to get infos of file '" << file_name << "'. \nSTOP." << endl;
        exit(EXIT_FAILURE);
    }

    Type *content = (Type *) calloc(info.st_size / sizeof(Type), sizeof(Type));
    assert (content != nullptr);

    FILE *fp = fopen(file_name.c_str(), "rb");
    if (fp == NULL) {
        cerr << "Unable to open file '" << file_name << "'. \nSTOP." << endl;
        exit(EXIT_FAILURE);
    }

    if (fread( (char *) content, info.st_size / sizeof(Type), 1, fp) != 1) {
        cerr << "Read failed for file '" << file_name << "'. \nSTOP." << endl;
        exit(EXIT_FAILURE);
    }

    fclose(fp);
    std::vector<Type> container (content, content  + info.st_size / sizeof(Type));
    container.shrink_to_fit();
    free (content);
    return container;
}

int main()
{
    std::string path_partkey_data, path_orderkey_data;
    if (true) {
        path_partkey_data = "/home/sebastian/cogadb_databases/cogadb_reference_databases_v1/cogadb_reference_databases/tpch_sf1/tables/LINEITEM/LINEITEM.L_PARTKEY.data";  // "/home/sebastian/cogadb_databases/tpch_sf10_new/tables/LINEITEM/LINEITEM.L_PARTKEY.data";
        path_orderkey_data = "/home/sebastian/cogadb_databases/cogadb_reference_databases_v1/cogadb_reference_databases/tpch_sf1/tables/LINEITEM/LINEITEM.L_ORDERKEY.data"; // "/home/sebastian/cogadb_databases/tpch_sf10_new/tables/LINEITEM/LINEITEM.L_ORDERKEY.data" ;
    } else {
        path_partkey_data =  "/Users/marcus/temp/dbsf10/LINEITEM.L_PARTKEY.data"; // "/Users/marcus/temp/databases/cogadb_reference_databases_v1/tpch_sf1/tables/LINEITEM/LINEITEM.L_PARTKEY.data";
        path_orderkey_data = "/Users/marcus/temp/dbsf10/LINEITEM.L_ORDERKEY.data"; // "/Users/marcus/temp/databases/cogadb_reference_databases_v1/tpch_sf1/tables/LINEITEM/LINEITEM.L_ORDERKEY.data";
    }

    get_column_file_path(&path_partkey_data, "L_PARTKEY");
    get_column_file_path(&path_orderkey_data, "L_ORDERKEY");

    cout << "Loading L_PARTKEY...";
    auto column_data_partkey = read_from_file<uint32_t>(path_partkey_data);
    cout << "DONE (" << column_data_partkey.size() << " elements).\n";

    cout << "Loading L_ORDERKEY...";
    auto column_data_orderkey = read_from_file<uint32_t>(path_orderkey_data);
    cout << "DONE (" << column_data_orderkey.size() << " elements).\n";

    assert (column_data_partkey.size() == column_data_orderkey.size());
    size_t num_elements = column_data_partkey.size();

    uint32_t *result_buffer = (uint32_t *) malloc (num_elements * sizeof(uint32_t));
    assert (result_buffer != nullptr);

    column<uint32_t> PARTKEY(column_data_partkey.data(), column_data_partkey.data() + num_elements);
    column<uint32_t> ORDERKEY(column_data_orderkey.data(), column_data_orderkey.data() + num_elements);

    cout << "Generate index on PARTKEY..." << std::flush;
    auto d = utils::profiling::measure<std::chrono::nanoseconds>::execute([&PARTKEY] () {
        PARTKEY.create_index();
    });
    cout << "DONE (" << d / 1000000.0f << " ms)" << endl;


    column_data_partkey.clear();

//    vpipes::functional::batched_materializes<uint32_t>::func_t materialize_from_orderkey = [&column_data_orderkey]
//            (uint32_t *out_begin, uint32_t *out_end, const size_t *begin, const size_t*end)
//    {
//        assert (out_end - out_begin >= end - begin);
//        uint32_t *data = column_data_orderkey.data();
//        size_t distance = (end - begin);
//
//        /* accessing the ORDERKEY column via raw data is a workaround since the
//         * column data structure does not support this opperation currently.
//         * Blame me for that, but it will be part of the vpipes query pipeline.
//         * Therefore, materialization will come in the later stages of the
//         * framework ;) */
//        POINTER_GATHER(out_begin, data, begin, (end - begin));
//    };

    double last_duration = 2e6;
    size_t last_filter_chunk_size = 0, last_scan_chunk_size = 0;
    size_t result_set_size = 0;
    const unsigned FILTER_CHUNK_SIZE_UPPER_BOUND = 600 * 4;
    const unsigned SCAN_CHUNK_SIZE_UPPER_BOUND = 600 * 10;

    for (unsigned filter_chunk_size = 10; filter_chunk_size < FILTER_CHUNK_SIZE_UPPER_BOUND; filter_chunk_size += 90) {
        for (unsigned scan_chunk_size = 10; scan_chunk_size < SCAN_CHUNK_SIZE_UPPER_BOUND; scan_chunk_size += 90) {
            long current_duration = 0;
            size_t num_samples = 3;
            result_set_size = 0;

            for (size_t i = 0; i < num_samples; i++) {
                vpipes::toolkit::no_operation<uint32_t> nop;
                using predicates = vpipes::functional::batched_predicates<uint32_t>;
                current_duration += utils::profiling::measure<std::chrono::nanoseconds>::execute(
                        [&PARTKEY, &nop, &scan_chunk_size, &filter_chunk_size]() {
                            auto table_scan = PARTKEY.table_scan(&nop,
                                                                 predicates::less_than::micro_optimized_impl(
                                                                         2000000,
                                                                         true),
                                                                 scan_chunk_size, filter_chunk_size);
                            table_scan->start();
                            free(table_scan);
                        });

            }

          //  if (result_set_size != 59986043)
          //      cerr << "WARNING: Result set size is unexcepted!" << endl;

            double current_duration_d = current_duration / double(num_samples) / 1000000.0f;
            cout << "avg duration: " << current_duration_d
                 << "ms @ / f-chunk size: " << filter_chunk_size
                 << "/ s-chunk size: " << scan_chunk_size
                 << ", best so far: " << last_duration
                 << "/ f-chunk size: " << last_filter_chunk_size
                 << "/ s-chunk size: " << last_scan_chunk_size
                 << " w/ result set size: " << result_set_size << endl;


            //     if (last_duration + 3 < current_duration_d) {
            //         cout << "best performance: " << last_duration << "ms @ chunk size: " << last_chunk_size << endl;
            //     }

            if (last_duration > current_duration_d) {
                last_duration = current_duration_d;
                last_filter_chunk_size = filter_chunk_size;
                last_scan_chunk_size = scan_chunk_size;
            }
        }



    }

    column_data_orderkey.clear();
    free (result_buffer);


    return EXIT_SUCCESS;
}