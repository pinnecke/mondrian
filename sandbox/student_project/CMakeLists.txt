cmake_minimum_required(VERSION 3.6)
project(student_project)

include_directories("../../pantheon/include")

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)
add_executable(student_project ${SOURCE_FILES})