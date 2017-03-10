cmake_minimum_required(VERSION 3.5)
project(mondrian)

find_package(Boost 1.57.0)
include_directories(${Boost_INCLUDE_DIRS})

if(Boost_FOUND)

    message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
    message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
    message(STATUS "Boost_VERSION: ${Boost_VERSION}")

    include_directories(${Boost_INCLUDE_DIRS})

endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS -O3)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories("include")
include_directories("../vpipes/include")

file(GLOB_RECURSE SOURCE_FILES
        "include/*.hpp"
        "include/*.cpp"
)

add_executable(
        mondrian ${SOURCE_FILES} main.cpp
)

add_executable(
        play_vpipes ${SOURCE_FILES} modules/playaround-query-engine/main.cpp
)

if(Boost_FOUND)

    target_link_libraries(mondrian ${Boost_LIBRARIES})

endif()