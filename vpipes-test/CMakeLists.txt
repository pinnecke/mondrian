cmake_minimum_required(VERSION 3.6)
project(tests)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS -O0)

#add_definitions(-DNWARNING)
#add_definitions(-DNINFO)

include_directories(
        "../vpipes/include"
)
include_directories(".")


# GTest
enable_testing()
find_package(GTest REQUIRED)

if(GTest_FOUND)
    message(STATUS "Google Test was found (version ${GTEST_VERSION})!")
    message(STATUS "\tinclude dir: ${GTEST_INCLUDE_DIRS}")
    message(STATUS "\tlibs dir: ${GTEST_BOTH_LIBRARIES}")
    include_directories(${GTEST_INCLUDE_DIRS})
endif()

if(GTest_FOUND)
    add_executable(VpipesTesting ./tester_main.cpp )
    target_link_libraries(VpipesTesting ${GTEST_BOTH_LIBRARIES} pthread)
endif()