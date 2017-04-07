//
// Created by Mahmoud Mohsen on 4/5/17.
//

#pragma once
#include <vpipes.hpp>

namespace mondrian {
    namespace vpipes
    {
        namespace datastructures
        {
            template<class elements_type>
            class datastructure{
            public:
                virtual void insert(const elements_type* element)=0;
                virtual bool contains(const elements_type* element)=0;
                virtual void erease_elements()=0;
            };
        }
    }
}
