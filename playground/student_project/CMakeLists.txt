cmake_minimum_required(VERSION 3.5)
project(query_engine)

include_directories("../../vpipes/include")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS -O3)

add_definitions(-DNINFO)

add_executable(task1 task1.cpp)
add_executable(task2 task2.cpp)
add_executable(task3 task3.cpp)
add_executable(task4 task4.cpp)
add_executable(task5 task5.cpp)
add_executable(task6 task6.cpp)
add_executable(task7 task7.cpp)
