cmake_minimum_required(VERSION 3.5)
project(predicates_playing)

include_directories("./vpipes/include")
include_directories("./utils")


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS -O3)

add_definitions(-DNINFO)

add_executable(MeasurePredicateTime ./measuring_time.cpp )
