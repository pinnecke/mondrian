cmake_minimum_required(VERSION 3.4)
project(gridstore)

include_directories("include")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pedantic -Wall \
    -Wno-missing-braces -Wextra -Wno-missing-field-initializers -Wformat=2 \
    -Wswitch-default -Wswitch-enum -Wcast-align -Wpointer-arith \
    -Wbad-function-cast -Wstrict-overflow=5 -Wstrict-prototypes -Winline \
    -Wundef -Wnested-externs -Wcast-qual -Wshadow -Wunreachable-code \
    -Wfloat-equal -Wstrict-aliasing=2 -Wredundant-decls \
    -Wold-style-definition \
    -ggdb3 -Wno-unused-parameter -Wno-unused-function \
    -O0 \
    -fno-omit-frame-pointer -fno-common -fstrict-aliasing")

set(SOURCE_FILES main.cxx include/Pantheon/Attribute.h core/Attribute.cxx include/Pantheon/error.h include/Pantheon/stddef.h include/Pantheon/schema.h core/error.cxx include/Pantheon/config.h include/Pantheon/strtable.h include/Pantheon/event.h core/event.cxx include/Pantheon/macros.h include/Pantheon/Container/Queue.h include/Pantheon/stdinc.h core/collections/Queue.cxx include/Pantheon/Container/RecycleBuffer.h core/collections/RecycleBuffer.cxx include/Pantheon/Container/Stripe.h core/config.cxx)
add_executable(gridstore ${SOURCE_FILES})