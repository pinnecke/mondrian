//
// Created by Mahmoud Mohsen on 3/15/17.
//
#pragma  once

#include <vpipes.hpp>

namespace utilities
{
    mondrian::vpipes::point_copy<size_t >::func_t ids_copier = [] (size_t *values, const size_t *tupletids, size_t num_of_ids)
    {
        for (auto i = 0; i< num_of_ids; ++i) {
            *(values+i) = *(tupletids+i);

        }
    };
}

std::size_t *create_column(std::size_t num_of_elements, bool fill_with_fives = true, bool fill = true)
{
    auto result = (std::size_t *) malloc (num_of_elements * sizeof(size_t));
    if (fill) {
        for (auto i = 0; i < num_of_elements; ++i)
            result[i] = fill_with_fives ?5 :i;
    }
    return result;
}

void delete_column(size_t *column)
{
    free (column);
}



    bool has_same_vals(std::size_t *input , std::size_t  *output, size_t num_elements){
        for (auto i = 0 ; i<num_elements ;i++  ){
            if (input[i]!=output[i]) return false;
        }
        return true;
    }


size_t* generate_expected_result (size_t input_length ,  std::function< bool(int) > pred_logic  ){
    auto expected_result = create_column(input_length, false);
    auto expected_result_start = expected_result;
    auto expected_result_end = expected_result+input_length;
    std::remove_if(expected_result_start,expected_result_end,pred_logic);

    return expected_result;
}