//
// Created by Mahmoud Mohsen on 4/5/17.
//

#pragma once
#include <vpipes.hpp>
#include <iostream>
#include <map>

namespace mondrian {
    namespace vpipes
    {
        namespace datastructures
        {
            template<class elements_type>
            class red_black_tree : public datastructure<elements_type>
            {
            private:
                std::map<elements_type,bool> elements_tree;

            public:
                red_black_tree()
                {

                }

            public:
                inline virtual void insert(const elements_type* element) override final __attribute__((always_inline))
                {
                                         elements_tree[ *element  ] = 1;
                }

                inline virtual bool contains(const elements_type* element) override final __attribute__((always_inline))
                {
                                         return   elements_tree[*element] ;
                }

                inline virtual void erease_elements() override final __attribute__((always_inline))
                {
                                        elements_tree.clear();
                }
            };
        }
    }
}
