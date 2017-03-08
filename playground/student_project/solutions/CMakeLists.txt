cmake_minimum_required(VERSION 3.5)
project(query_engine)

include_directories("vpipes/include")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS -O3)

add_definitions(-DNINFO)

add_executable(task1-a task1-a.cpp)
add_executable(task1-b task1-b.cpp)
add_executable(task1-c task1-c.cpp)
add_executable(task1-e-g task1-e-g.cpp)
add_executable(task1-i task1-i.cpp)
add_executable(task3 task3.cpp)

