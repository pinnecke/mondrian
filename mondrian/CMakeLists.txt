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
include_directories("include")

file(GLOB_RECURSE SOURCE_FILES
        "include/*.hpp"
        "include/*.cpp"
)

add_executable(
        mondrian ${SOURCE_FILES} main.cpp
)

if(Boost_FOUND)

    target_link_libraries(mondrian ${Boost_LIBRARIES})

endif()