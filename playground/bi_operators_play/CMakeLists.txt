cmake_minimum_required(VERSION 3.6)

project(bi_operators_play)

set(CMAKE_CXX_STANDARD 14)



include_directories(../../vpipes/include)
include_directories(../../vpipes/include/vpipes/pipes/joins/hash_libs/sparsehash)
set(SOURCE_FILES stl_un_ordered_hash_join.cpp)
add_executable(stl_un_ordered_hash_join_play ${SOURCE_FILES})
add_executable(block_nested_join_play block_nested_join_play.cpp)
add_executable(cuckoo_play cuckoo_playing.cpp)
add_executable(dense_hash_playing dense_hash_play.cpp)
add_executable(sparse_hash_playing sparse_hash_play.cpp)

target_link_libraries(cuckoo_play pthread)

