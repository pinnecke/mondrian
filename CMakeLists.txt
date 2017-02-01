cmake_minimum_required(VERSION 3.4)
project(gridstore)

include_directories("include")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -pedantic -Wall \
    -Wno-missing-braces -Wextra -Wno-missing-field-initializers -Wformat=2 \
    -Wswitch-default -Wswitch-enum -Wcast-align -Wpointer-arith \
    -Wbad-function-cast -Wstrict-overflow=5 -Wstrict-prototypes -Winline \
    -Wundef -Wnested-externs -Wcast-qual -Wshadow -Wunreachable-code \
    -Wfloat-equal -Wstrict-aliasing=2 -Wredundant-decls \
    -Wold-style-definition \
    -ggdb3 -Wno-unused-parameter -Wno-unused-function \
    -O0 \
    -fno-omit-frame-pointer -fno-common -fstrict-aliasing")

set(SOURCE_FILES include/Pantheon/Attribute.h Pantheon/Attribute.cxx include/Pantheon/error.h include/Pantheon/stddef.h include/Pantheon/schema.h Pantheon/error.cxx include/Pantheon/config.h include/Pantheon/strtable.h include/Pantheon/event.h Pantheon/event.cxx include/Pantheon/macros.h include/Pantheon/Container/Queue.h include/Pantheon/stdinc.h Pantheon/Container/Queue.cxx include/Pantheon/Container/RecycleBuffer.h Pantheon/Container/RecycleBuffer.cxx include/Pantheon/Container/Stripe.h Pantheon/config.cxx Experiments/PagingVsBuffering/launcher.cpp include/Pantheon/IO/BufferManager.h Pantheon/IO/BufferManager.cxx include/Pantheon/Container/ArrayList.h Pantheon/Container/ArrayList.cxx include/Pantheon/Container/LinkedList.h Pantheon/Container/LinkedList.cxx)

set(MAIN_LAUNCER main.cxx ${SOURCE_FILES})
set(EXP_LAUNCER Experiments/PagingVsBuffering/launcher.cpp ${SOURCE_FILES} )
set(EXP_LAUNCER_ROW_VS_COLUMN_STORE Experiments/RowVsColumnStore/launcher.cpp ${SOURCE_FILES} )


add_executable(gridstore ${MAIN_LAUNCER})
add_executable(exp ${EXP_LAUNCER})
add_executable(exp_row_vs_column_store Experiments/RowVsColumnStore/launcher.cpp)