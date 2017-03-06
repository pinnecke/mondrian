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
include_directories("include", "include/utils", "include/modules")

file(GLOB_RECURSE SOURCE_FILES
        "include/*.hpp"
        "include/*.cpp"
)

set(SOURCE_FILES
        include/modules/modules.h
        include/utils/result.h
        modules/modules.c
        include/utils/kernel.h
        modules/version/main.c
        include/modules/version/main.h
        include/utils/alg.h
        utils/alg.c include/utils/debug_build.h)

add_executable(
        mondrian ${SOURCE_FILES} main.cpp
        include/modules/modules.h include/modules/version/main.h modules/version/main.c include/modules/modules.h include/modules/manifest.h include/utils/result.h include/utils/alg.h modules/modules.c utils/alg.c include/utils/kernel.h include/utils/sconfig.h include/utils/log.h include/utils/ntsid.h)

if(Boost_FOUND)

    target_link_libraries(mondrian ${Boost_LIBRARIES})

endif()