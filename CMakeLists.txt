cmake_minimum_required(VERSION 3.4)
project(gridstore)

include_directories("include")

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c11 -Wall -Werror")

set(SOURCE_FILES main.c include/core/attribute.h core/attribute.c include/core/error.h include/core/stddef.h include/core/schema.h core/error.c include/core/config.h core/config.c include/core/strtable.h include/core/event.h core/event.c include/core/mtb.h include/core/collections/queue.h include/core/stdinc.h core/collections/queue.c include/core/collections/recycle_buffer.h)
add_executable(gridstore ${SOURCE_FILES})