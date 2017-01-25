cmake_minimum_required(VERSION 3.4)
project(gridstore)

include_directories("include")

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c11 -pedantic -Wall \
    -Wno-missing-braces -Wextra -Wno-missing-field-initializers -Wformat=2 \
    -Wswitch-default -Wswitch-enum -Wcast-align -Wpointer-arith \
    -Wbad-function-cast -Wstrict-overflow=5 -Wstrict-prototypes -Winline \
    -Wundef -Wnested-externs -Wcast-qual -Wshadow -Wunreachable-code \
    -Wlogical-op -Wfloat-equal -Wstrict-aliasing=2 -Wredundant-decls \
    -Wold-style-definition -Werror \
    -ggdb3 \
    -O0 \
    -fno-omit-frame-pointer -ffloat-store -fno-common -fstrict-aliasing \
    -lm")

set(SOURCE_FILES main.c include/core/attribute.h core/attribute.c include/core/error.h include/core/stddef.h include/core/schema.h core/error.c include/core/config.h core/config.c include/core/strtable.h include/core/event.h core/event.c include/core/macros.h include/core/collections/queue.h include/core/stdinc.h core/collections/queue.c include/core/collections/recycle_buffer.h core/collections/recycle_buffer.c include/core/collections/stripe.h)
add_executable(gridstore ${SOURCE_FILES})