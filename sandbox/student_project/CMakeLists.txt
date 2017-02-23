cmake_minimum_required(VERSION 3.5)
project(query_engine)

include_directories("../../pantheon/include")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS -O3)

set(SOURCE_FILES main.cpp)
add_executable(query_engine ${SOURCE_FILES})
