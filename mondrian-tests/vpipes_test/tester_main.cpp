//
// Created by Mahmoud Mohsen on 3/15/17.
//

#include "gtest/gtest.h"
#include <vpipes.hpp>
#include <testing_chunks.hpp>

#include <testing_reader_materlizer.hpp>

#include <testing_filters.hpp>
#include <testing_table_scan.hpp>

 using namespace std;
using namespace mondrian::vpipes;


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}