cmake_minimum_required(VERSION 3.5)
project(query_engine)

include_directories("../../mondrian/include")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS -O3)

add_definitions(-DNINFO)

set(SOURCE_FILES main.cpp)
add_executable(query_engine ${SOURCE_FILES})
