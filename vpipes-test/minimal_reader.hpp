//
// Created by Mahmoud Mohsen on 3/22/17.
//

#pragma once

#include <vpipes.hpp>
#include <testing_utilities.hpp>


using namespace mondrian::vpipes;

namespace testing_vpipes_classes{

    template <class ValueType>
    class minimal_reader
    {
    public:
        using value_t = ValueType;
        using tupletid_t = size_t;
        using table_scan_t = pipes::table_scan<ValueType>;
        using predicate_t = typename table_scan_t::predicate_t;
        using point_copy_t = typename point_copy<value_t, size_t>::func_t;
    private:
        consumer<value_t> *m_consumer;
        predicate_t m_predicate;
        unsigned m_scan_chunk_size;
        unsigned m_filter_chunk_size;
        size_t   m_total_elements;
    public:
        minimal_reader(consumer<value_t> *consumer_p, predicate_t predicate,
                       size_t total_elements,unsigned scan_chunk_size ,unsigned filter_chunk_size):m_consumer(consumer_p),
                                                                                                   m_predicate(predicate),
                                                                                                   m_total_elements(total_elements),
                                                                                                   m_scan_chunk_size(scan_chunk_size),
                                                                                                   m_filter_chunk_size(filter_chunk_size)
        {

        }

        inline virtual size_t *  materlializer(){
            return  create_column(m_total_elements,false);
        }

        inline virtual void read() final __attribute__((always_inline))
        {
            size_t start = 0, end = m_total_elements;
            interval<size_t> all_tuplet_ids(start, end);

            mondrian::vpipes::block_copy<size_t>::func_t loc_block_copy = [](size_t *out, size_t begin, size_t end){
                auto distance = end- begin;
                for (auto i = 0 ; i<distance ;++i){
                    *(out+i) =begin+i;
                }
            };

            auto loc_table = pipes::table_scan<value_t>(m_consumer, &all_tuplet_ids, &all_tuplet_ids + 1, m_predicate,
                                                        loc_block_copy   , m_scan_chunk_size, m_filter_chunk_size);

            loc_table.start();
        }
    };

}

