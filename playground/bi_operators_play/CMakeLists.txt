cmake_minimum_required(VERSION 3.6)

project(bi_operators_play)

set(CMAKE_CXX_STANDARD 14)

include_directories(../../vpipes/include)

set(SOURCE_FILES main.cpp)
add_executable(bi_operators_play ${SOURCE_FILES})