//
// Created by Mahmoud Mohsen on 3/15/17.
//
#pragma  once



#define PREDICATE(x)                                                                    \
[] (size_t *result, size_t *result_size, size_t const *begin, size_t const *end,size_t const *values_begin, size_t const *values_end) {                       \
    size_t i = 0;                                                                       \
        \
    for (auto it = values_begin; it != values_end; ++it) {                                            \
        if (*it % x == 0)                                                              \
            (result)[i++] = *it;                                                       \
    }                                                                                   \
    *result_size = i;                                                                   \
}

#define PREDICATE2(x)                                                                    \
[] (size_t *result, size_t *result_size, size_t const *begin, size_t const *end,size_t const *values_begin, size_t const *values_end) {                       \
    size_t i = 0;                                                                       \
        \
    for (auto it = values_begin; it != values_end; ++it) {                                            \
        if (*it > x)                                                              \
            (result)[i++] = *it;                                                       \
    }                                                                                   \
    *result_size = i;                                                                   \
}





template <class InputType>

std::size_t *create_column(unsigned long num_of_elements, bool fill_with_fives = true, bool fill = true)
{
    auto result = (std::size_t *) malloc (num_of_elements * sizeof(InputType));
    if (fill) {
        for (auto i = 0; i < num_of_elements; ++i)
            result[i] = fill_with_fives ?5 :i;
    }
    return result;
}

template <class InputType>
void delete_column(InputType *column)
{
    free (column);
}



template <class dtype>
    bool has_same_vals(dtype input, dtype output, size_t num_elements){
    for (auto i = 0 ; i<num_elements ;i++  ){
        if (input[i]!=output[i]) return false;
    }
    return true;
}


vector <size_t> generate_vector_from_intervals( size_t *begins , size_t *ends, size_t num_intervals, int predicate   ){
    vector <size_t> generated_vec ;
    for (auto i =0 ; i <num_intervals ;i++){
        for (auto j = begins[i]; j<ends[i]; j++  ){
            if (j%predicate ==0){
                generated_vec.push_back(j);
            }
        }
    }

    return generated_vec;
}

