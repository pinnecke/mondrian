cmake_minimum_required(VERSION 3.6)
project(tests)

set(CMAKE_CXX_STANDARD 14)

include_directories(
        "../pantheon/include"
)

# Boost
find_package(Boost 1.57.0 REQUIRED)

if(Boost_FOUND)
    message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
    message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
    message(STATUS "Boost_VERSION: ${Boost_VERSION}")

    include_directories(${Boost_INCLUDE_DIRS})
endif()

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
    add_executable(foo storage/host_vector_column_test.cpp)
    target_link_libraries(foo ${GTEST_BOTH_LIBRARIES})
    target_link_libraries(foo ${Boost_LIBRARIES})
endif()