cmake_minimum_required(VERSION 3.7)
project(vpipes_play)

set(CMAKE_CXX_STANDARD 14)

include_directories("../../vpipes/include")


add_executable(main main.cpp)
add_executable(table_play play_with_table_scan.cpp)
add_executable(filter_play play_with_filters.cpp)
add_executable(project_play play_with_project.cpp)
add_executable(tee_play play_with_tee.cpp)
add_executable(map_play play_with_map.cpp)