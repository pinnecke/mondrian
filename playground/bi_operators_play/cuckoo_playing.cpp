#include <iostream>
#include <vpipes.hpp>
#include <cstring>
mondrian::vpipes::point_null_copy::func_t null_copier = [] (mondrian::mtl::smart_bitmask *mask, const size_t *tupletids, size_t num_of_ids)
{
    mask->unset_all();
};

int main() {


    using namespace mondrian::vpipes;
    size_t res_length = 1000;
    auto batch_size =4;
    auto  input_length= 8;
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

    mondrian::vpipes::pipes::tee <size_t> tee_op(&val_mat, &ids_mat, batch_size);

    mondrian::vpipes::bi_pipes::cuckoo_hash_join <size_t> b_join (&tee_op, batch_size);

    auto loc_table = pipes::table_scan<size_t >(b_join.get_inner_operand(), &all_tuplet_ids, &all_tuplet_ids + 1,mondrian::vpipes::predicates::batched_predicates<size_t >
                                                ::greater_equal::micro_optimized_impl(predicate_value,true),
                                                null_value_filter_policy::skip_null_values,
                                                loc_block_copy, loc_block_null_copy, batch_size, batch_size, true);

    auto loc_table2 = pipes::table_scan<size_t >(b_join.get_outer_operand(), &all_tuplet_ids, &all_tuplet_ids + 1,mondrian::vpipes::predicates::batched_predicates<size_t >
                                                 ::greater_equal::micro_optimized_impl(predicate_value,true),
                                                 null_value_filter_policy::skip_null_values,
                                                 loc_block_copy2, loc_block_null_copy, batch_size, batch_size, true);

    loc_table.start();
    loc_table2.start();

    for (auto i =0 ; i < res_length ;++i){
        std::cout<<"id ="<<ids_result[i]<<" , "<< val_result[i] <<std::endl;
    }

    std::cout<<"num of null branches = "<<b_join.get_join_statistics()->count_null_branch_used<<std::endl;
    std::cout<<"num of none null branches = "<<b_join.get_join_statistics()->count_non_null_branch_used<<std::endl;
    std::cout<<"num of count null values = "<<b_join.get_join_statistics()->count_null_values<<std::endl;
    std::cout<<"count join pairs = "<<b_join.get_join_statistics()->count_join_pairs<<std::endl;

    delete []val_result;
    delete []ids_result;
    return 0;
}