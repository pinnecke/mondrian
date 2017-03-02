cmake_minimum_required(VERSION 3.5)
project(vecpips)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS -O3)

set(SOURCES include/producer.hpp)
add_library(vecpipes STATIC ${SOURCES})
SET_TARGET_PROPERTIES(vecpipes PROPERTIES OUTPUT_NAME mylib CLEAN_DIRECT_OUTPUT 1)

