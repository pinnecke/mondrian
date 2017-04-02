//
// Created by Mahmoud Mohsen on 3/31/17.
//
#include <random>
#pragma once

namespace  random_gen {
    size_t *random_container;
    default_random_engine generator;
    size_t num_elements;
    auto lower_bound =0;
    auto higher_bound =50 ;
    bool reset = false;
    bool filled = false;


    size_t random_number()
    {
        static uniform_int_distribution<int > distribution(lower_bound,higher_bound);
        return distribution(generator);
    };

    void allocate_random_container (size_t elements){
              num_elements =elements;
              random_container =  new size_t[num_elements];
    }

    void delete_random_container(){
        delete [] random_container;
    }


     size_t  gen_random_number() {

        static auto idx =0 ;
        if (idx >= num_elements||reset) {
            idx =0 ;
            reset = false;
        }
        if (! filled){
            random_container[idx] =random_number();
        }
        auto res = random_container[idx];
        ++idx;
        return res;
    }


}


