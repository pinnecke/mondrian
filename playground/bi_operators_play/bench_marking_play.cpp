#include <iostream>
#include <vpipes.hpp>
#include <cstring>
#include "profiler.hpp"
#include <iomanip>
#include "fstream"

mondrian::vpipes::point_null_copy::func_t null_copier = [] (mondrian::mtl::smart_bitmask *mask, const size_t *tupletids, size_t num_of_ids)
{
    mask->unset_all();
};


#define PATH_TO_RESULTS_FILE "/home/pegasus/results/joins_result_files.csv"

int main() {

    using namespace mondrian::vpipes;

    std::ofstream results_file(PATH_TO_RESULTS_FILE);
    if (results_file.is_open()) {
        cout <<"measuring for measuring for cuckoo join"<<">>>>>>>>>>>>>>>>"<<std::endl;

        // measuring for cuckoo join
        for (size_t input_length =100 ; input_length <= 1000000; input_length *= 10 ){
            size_t res_length = input_length;
            auto batch_size =4;
            auto  val_result = new size_t[res_length];
            auto  ids_result = new size_t[res_length];
            auto predicate_value = 0 ;

            mondrian::vpipes::point_copy<size_t >::func_t ids_copier = [] (size_t *values, const size_t *tupletids, size_t num_of_ids)
            {
                for (auto i = 0; i< num_of_ids; ++i) {
                    *(values+i) = *(tupletids+i);

                }
            };
            mondrian::vpipes::pipes::val_materialize<size_t> val_mat(val_result, &res_length);
            mondrian::vpipes::pipes::tid_materialize<size_t> ids_mat(ids_result, &res_length);

            interval<size_t> all_tuplet_ids(0, input_length);


            mondrian::vpipes::block_copy<size_t>::func_t loc_block_copy = [](size_t *values, size_t begin, size_t end){
                auto distance = end- begin;
                for (auto i = 0 ; i<distance ;++i){
                    *(values+i) =begin+i;
                }
            };

            mondrian::vpipes::block_copy<size_t>::func_t loc_block_copy2 = [](size_t *values, size_t begin, size_t end){
                auto distance = end- begin;
                for (auto i = 0 ; i<distance ;++i){
                    *(values+i) =7;
                }
            };

            mondrian::vpipes::block_null_copy::func_t loc_block_null_copy = [] (mondrian::mtl::smart_bitmask *values,
                                                                                const mondrian::mtl::smart_array<size_t> *null_mask_indices,
                                                                                const mondrian::mtl::smart_array<tuplet_id_t> *tuplet_ids)
            {
                assert (values != nullptr);
                values->unset_range_safe(0, null_mask_indices->get_num_elements());
            };

            mondrian::vpipes::block_null_copy::func_t loc_block_null_copy2 = [] (mondrian::mtl::smart_bitmask *null_mask,
                                                                                 const mondrian::mtl::smart_array<size_t> *null_mask_indices,
                                                                                 const mondrian::mtl::smart_array<tuplet_id_t> *tuplet_ids)
            {
                for (int i = 0; i < null_mask_indices->get_num_elements(); i += 2) {
                    null_mask->set_unsafe(i, true);
                    assert (null_mask->get_unsafe(i));
                }
            };



            size_t current_duration = 0;


            mondrian::vpipes::pipes::tee <size_t> tee_op(&val_mat, &ids_mat, batch_size);


            mondrian::vpipes::bi_pipes::cuckoo_hash_join <size_t> cuckoo_join (&tee_op, batch_size);

            auto loc_table = pipes::table_scan<size_t >(cuckoo_join.get_inner_operand(), &all_tuplet_ids, &all_tuplet_ids + 1,mondrian::vpipes::predicates::batched_predicates<size_t >
                                                        ::greater_equal::micro_optimized_impl(predicate_value,true),
                                                        null_value_filter_policy::skip_null_values,
                                                        loc_block_copy, loc_block_null_copy, batch_size, batch_size, true);

            auto loc_table2 = pipes::table_scan<size_t >(cuckoo_join.get_outer_operand(), &all_tuplet_ids, &all_tuplet_ids + 1,mondrian::vpipes::predicates::batched_predicates<size_t >
                                                         ::greater_equal::micro_optimized_impl(predicate_value,true),
                                                         null_value_filter_policy::skip_null_values,
                                                         loc_block_copy, loc_block_null_copy, batch_size, batch_size, true);

            current_duration = mondrian::utils::profiling::measure<std::chrono::nanoseconds>::execute(
                    [&tee_op,&cuckoo_join, &loc_table, &loc_table2]() {

                        loc_table.start();
                        loc_table2.start();

                    });
            double current_duration_d = (current_duration ) / 1000000.0f;
            results_file << std::fixed << std::setprecision(1) << input_length << "," << current_duration_d << "," << "cuckoo_based_hash" << "\n";

            cuckoo_join.clean_storage();
            delete []val_result;
            delete []ids_result;
        }
        cout <<"measuring for measuring for stl join"<<">>>>>>>>>>>>>>>>"<<std::endl;
        // measuring for stl join
        for (size_t input_length =100 ; input_length <= 1000000; input_length *= 10 ){
            size_t res_length = input_length;
            auto batch_size =4;
            auto  val_result = new size_t[res_length];
            auto  ids_result = new size_t[res_length];
            auto predicate_value = 0 ;

            mondrian::vpipes::point_copy<size_t >::func_t ids_copier = [] (size_t *values, const size_t *tupletids, size_t num_of_ids)
            {
                for (auto i = 0; i< num_of_ids; ++i) {
                    *(values+i) = *(tupletids+i);

                }
            };
            mondrian::vpipes::pipes::val_materialize<size_t> val_mat(val_result, &res_length);
            mondrian::vpipes::pipes::tid_materialize<size_t> ids_mat(ids_result, &res_length);

            interval<size_t> all_tuplet_ids(0, input_length);


            mondrian::vpipes::block_copy<size_t>::func_t loc_block_copy = [](size_t *values, size_t begin, size_t end){
                auto distance = end- begin;
                for (auto i = 0 ; i<distance ;++i){
                    *(values+i) =begin+i;
                }
            };

            mondrian::vpipes::block_copy<size_t>::func_t loc_block_copy2 = [](size_t *values, size_t begin, size_t end){
                auto distance = end- begin;
                for (auto i = 0 ; i<distance ;++i){
                    *(values+i) =7;
                }
            };

            mondrian::vpipes::block_null_copy::func_t loc_block_null_copy = [] (mondrian::mtl::smart_bitmask *values,
                                                                                const mondrian::mtl::smart_array<size_t> *null_mask_indices,
                                                                                const mondrian::mtl::smart_array<tuplet_id_t> *tuplet_ids)
            {
                assert (values != nullptr);
                values->unset_range_safe(0, null_mask_indices->get_num_elements());
            };

            mondrian::vpipes::block_null_copy::func_t loc_block_null_copy2 = [] (mondrian::mtl::smart_bitmask *null_mask,
                                                                                 const mondrian::mtl::smart_array<size_t> *null_mask_indices,
                                                                                 const mondrian::mtl::smart_array<tuplet_id_t> *tuplet_ids)
            {
                for (int i = 0; i < null_mask_indices->get_num_elements(); i += 2) {
                    null_mask->set_unsafe(i, true);
                    assert (null_mask->get_unsafe(i));
                }
            };



            size_t current_duration = 0;


            mondrian::vpipes::pipes::tee <size_t> tee_op(&val_mat, &ids_mat, batch_size);


            mondrian::vpipes::bi_pipes::stl_hash_join <size_t> stl_based (&tee_op, batch_size);

            auto loc_table = pipes::table_scan<size_t >(stl_based.get_inner_operand(), &all_tuplet_ids, &all_tuplet_ids + 1,mondrian::vpipes::predicates::batched_predicates<size_t >
                                                        ::greater_equal::micro_optimized_impl(predicate_value,true),
                                                        null_value_filter_policy::skip_null_values,
                                                        loc_block_copy, loc_block_null_copy, batch_size, batch_size, true);

            auto loc_table2 = pipes::table_scan<size_t >(stl_based.get_outer_operand(), &all_tuplet_ids, &all_tuplet_ids + 1,mondrian::vpipes::predicates::batched_predicates<size_t >
                                                         ::greater_equal::micro_optimized_impl(predicate_value,true),
                                                         null_value_filter_policy::skip_null_values,
                                                         loc_block_copy, loc_block_null_copy, batch_size, batch_size, true);

            current_duration = mondrian::utils::profiling::measure<std::chrono::nanoseconds>::execute(
                    [&tee_op,&stl_based, &loc_table, &loc_table2]() {

                        loc_table.start();
                        loc_table2.start();

                    });
            double current_duration_d = (current_duration ) / 1000000.0f;
            results_file << std::fixed << std::setprecision(1) << input_length << "," << current_duration_d << "," << "stl_hash_join" << "\n";

            stl_based.clean_storage();
            delete []val_result;
            delete []ids_result;
        }
        cout <<"measuring for measuring for sparse join"<<">>>>>>>>>>>>>>>>"<<std::endl;
        // measuring for sparse join
        for (size_t input_length =100 ; input_length <= 1000000; input_length *= 10 ){
            size_t res_length = input_length;
            auto batch_size =4;
            auto  val_result = new size_t[res_length];
            auto  ids_result = new size_t[res_length];
            auto predicate_value = 0 ;

            mondrian::vpipes::point_copy<size_t >::func_t ids_copier = [] (size_t *values, const size_t *tupletids, size_t num_of_ids)
            {
                for (auto i = 0; i< num_of_ids; ++i) {
                    *(values+i) = *(tupletids+i);

                }
            };
            mondrian::vpipes::pipes::val_materialize<size_t> val_mat(val_result, &res_length);
            mondrian::vpipes::pipes::tid_materialize<size_t> ids_mat(ids_result, &res_length);

            interval<size_t> all_tuplet_ids(0, input_length);


            mondrian::vpipes::block_copy<size_t>::func_t loc_block_copy = [](size_t *values, size_t begin, size_t end){
                auto distance = end- begin;
                for (auto i = 0 ; i<distance ;++i){
                    *(values+i) =begin+i;
                }
            };

            mondrian::vpipes::block_copy<size_t>::func_t loc_block_copy2 = [](size_t *values, size_t begin, size_t end){
                auto distance = end- begin;
                for (auto i = 0 ; i<distance ;++i){
                    *(values+i) =7;
                }
            };

            mondrian::vpipes::block_null_copy::func_t loc_block_null_copy = [] (mondrian::mtl::smart_bitmask *values,
                                                                                const mondrian::mtl::smart_array<size_t> *null_mask_indices,
                                                                                const mondrian::mtl::smart_array<tuplet_id_t> *tuplet_ids)
            {
                assert (values != nullptr);
                values->unset_range_safe(0, null_mask_indices->get_num_elements());
            };

            mondrian::vpipes::block_null_copy::func_t loc_block_null_copy2 = [] (mondrian::mtl::smart_bitmask *null_mask,
                                                                                 const mondrian::mtl::smart_array<size_t> *null_mask_indices,
                                                                                 const mondrian::mtl::smart_array<tuplet_id_t> *tuplet_ids)
            {
                for (int i = 0; i < null_mask_indices->get_num_elements(); i += 2) {
                    null_mask->set_unsafe(i, true);
                    assert (null_mask->get_unsafe(i));
                }
            };



            size_t current_duration = 0;


            mondrian::vpipes::pipes::tee <size_t> tee_op(&val_mat, &ids_mat, batch_size);


            mondrian::vpipes::bi_pipes::sparse_map_hash_join <size_t> sparse_based (&tee_op, batch_size);

            auto loc_table = pipes::table_scan<size_t >(sparse_based.get_inner_operand(), &all_tuplet_ids, &all_tuplet_ids + 1,mondrian::vpipes::predicates::batched_predicates<size_t >
                                                        ::greater_equal::micro_optimized_impl(predicate_value,true),
                                                        null_value_filter_policy::skip_null_values,
                                                        loc_block_copy, loc_block_null_copy, batch_size, batch_size, true);

            auto loc_table2 = pipes::table_scan<size_t >(sparse_based.get_outer_operand(), &all_tuplet_ids, &all_tuplet_ids + 1,mondrian::vpipes::predicates::batched_predicates<size_t >
                                                         ::greater_equal::micro_optimized_impl(predicate_value,true),
                                                         null_value_filter_policy::skip_null_values,
                                                         loc_block_copy, loc_block_null_copy, batch_size, batch_size, true);

            current_duration = mondrian::utils::profiling::measure<std::chrono::nanoseconds>::execute(
                    [&tee_op,&sparse_based, &loc_table, &loc_table2]() {

                        loc_table.start();
                        loc_table2.start();

                    });
            double current_duration_d = (current_duration ) / 1000000.0f;
            results_file << std::fixed << std::setprecision(1) << input_length << "," << current_duration_d << "," << "sparse_hash_join" << "\n";

            sparse_based.clean_storage();
            delete []val_result;
            delete []ids_result;
        }
        cout <<"measuring for measuring for dense join"<<">>>>>>>>>>>>>>>>"<<std::endl;
        // measuring for dense join
        for (size_t input_length =100 ; input_length <= 1000000; input_length *= 10 ){
            size_t res_length = input_length;
            auto batch_size =4;
            auto  val_result = new size_t[res_length];
            auto  ids_result = new size_t[res_length];
            auto predicate_value = 0 ;

            mondrian::vpipes::point_copy<size_t >::func_t ids_copier = [] (size_t *values, const size_t *tupletids, size_t num_of_ids)
            {
                for (auto i = 0; i< num_of_ids; ++i) {
                    *(values+i) = *(tupletids+i);

                }
            };
            mondrian::vpipes::pipes::val_materialize<size_t> val_mat(val_result, &res_length);
            mondrian::vpipes::pipes::tid_materialize<size_t> ids_mat(ids_result, &res_length);

            interval<size_t> all_tuplet_ids(0, input_length);


            mondrian::vpipes::block_copy<size_t>::func_t loc_block_copy = [](size_t *values, size_t begin, size_t end){
                auto distance = end- begin;
                for (auto i = 0 ; i<distance ;++i){
                    *(values+i) =begin+i;
                }
            };

            mondrian::vpipes::block_copy<size_t>::func_t loc_block_copy2 = [](size_t *values, size_t begin, size_t end){
                auto distance = end- begin;
                for (auto i = 0 ; i<distance ;++i){
                    *(values+i) =7;
                }
            };

            mondrian::vpipes::block_null_copy::func_t loc_block_null_copy = [] (mondrian::mtl::smart_bitmask *values,
                                                                                const mondrian::mtl::smart_array<size_t> *null_mask_indices,
                                                                                const mondrian::mtl::smart_array<tuplet_id_t> *tuplet_ids)
            {
                assert (values != nullptr);
                values->unset_range_safe(0, null_mask_indices->get_num_elements());
            };

            mondrian::vpipes::block_null_copy::func_t loc_block_null_copy2 = [] (mondrian::mtl::smart_bitmask *null_mask,
                                                                                 const mondrian::mtl::smart_array<size_t> *null_mask_indices,
                                                                                 const mondrian::mtl::smart_array<tuplet_id_t> *tuplet_ids)
            {
                for (int i = 0; i < null_mask_indices->get_num_elements(); i += 2) {
                    null_mask->set_unsafe(i, true);
                    assert (null_mask->get_unsafe(i));
                }
            };



            size_t current_duration = 0;


            mondrian::vpipes::pipes::tee <size_t> tee_op(&val_mat, &ids_mat, batch_size);


            mondrian::vpipes::bi_pipes::dense_map_hash_join <size_t> dense_based (&tee_op, batch_size, input_length);

            auto loc_table = pipes::table_scan<size_t >(dense_based.get_inner_operand(), &all_tuplet_ids, &all_tuplet_ids + 1,mondrian::vpipes::predicates::batched_predicates<size_t >
                                                        ::greater_equal::micro_optimized_impl(predicate_value,true),
                                                        null_value_filter_policy::skip_null_values,
                                                        loc_block_copy, loc_block_null_copy, batch_size, batch_size, true);

            auto loc_table2 = pipes::table_scan<size_t >(dense_based.get_outer_operand(), &all_tuplet_ids, &all_tuplet_ids + 1,mondrian::vpipes::predicates::batched_predicates<size_t >
                                                         ::greater_equal::micro_optimized_impl(predicate_value,true),
                                                         null_value_filter_policy::skip_null_values,
                                                         loc_block_copy, loc_block_null_copy, batch_size, batch_size, true);

            current_duration = mondrian::utils::profiling::measure<std::chrono::nanoseconds>::execute(
                    [&tee_op, &dense_based, &loc_table, &loc_table2]() {

                        loc_table.start();
                        loc_table2.start();

                    });
            double current_duration_d = (current_duration ) / 1000000.0f;
            results_file << std::fixed << std::setprecision(1) << input_length << "," << current_duration_d << "," << "dense_hash_join" << "\n";

            dense_based.clean_storage();
            delete []val_result;
            delete []ids_result;
        }

    }


    return 0;
}