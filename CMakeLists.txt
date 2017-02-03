cmake_minimum_required(VERSION 3.4)

project(gridstore)

FIND_PACKAGE(CUDA REQUIRED)


include_directories("include")

#set(
#        CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -pthread -pedantic -Wall"
#)

set(
        SOURCE_FILES
        include/Pantheon/Attribute.h
        Pantheon/Attribute.cxx
        include/Pantheon/error.h
        include/Pantheon/stddef.h
        include/Pantheon/schema.h
        Pantheon/error.cxx
        include/Pantheon/config.h
        include/Pantheon/strtable.h
        include/Pantheon/event.h
        Pantheon/event.cxx
        include/Pantheon/macros.h
        include/Pantheon/Container/Queue.h
        include/Pantheon/stdinc.h
        Pantheon/Container/Queue.cxx
        include/Pantheon/Container/RecycleBuffer.h
        Pantheon/Container/RecycleBuffer.cxx
        include/Pantheon/Container/Stripe.h
        Pantheon/config.cxx
        Experiments/PagingVsBuffering/launcher.cpp
        include/Pantheon/IO/BufferManager.h
        Pantheon/IO/BufferManager.cxx
        include/Pantheon/Container/ArrayList.h
        Pantheon/Container/ArrayList.cxx
        include/Pantheon/Container/LinkedList.h
        Pantheon/Container/LinkedList.cxx
)

set(
        MAIN_LAUNCER
        main.cxx ${SOURCE_FILES}
)

set(
        EXP_LAUNCER
        Experiments/PagingVsBuffering/launcher.cpp
        ${SOURCE_FILES}
)

set(
        EXP_LAUNCER_ROW_VS_COLUMN_STORE
        Experiments/RowVsColumnStore/HostLauncher.cu
        Experiments/RowVsColumnStore/Configure.h
        Experiments/RowVsColumnStore/Host/Query.h
        Experiments/RowVsColumnStore/Host/Query.cxx
        Experiments/RowVsColumnStore/Shared/Common.h
        Experiments/RowVsColumnStore/Shared/Common.cpp
        Experiments/RowVsColumnStore/Shared/Utility.h
        Experiments/RowVsColumnStore/Host/Parallel.h Experiments/RowVsColumnStore/Host/Operations.h)


add_executable(gridstore ${MAIN_LAUNCER})
add_executable(exp ${EXP_LAUNCER})

INCLUDE(FindCUDA)

include_directories(${CUDA_INCLUDE_DIRS})

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11 -m64 -gencode arch=compute_35,code=sm_35)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_35,code=sm_35;--std=c++11)
set(CUDA_HOST_COMPILER clang++)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

CUDA_ADD_EXECUTABLE(exp-row-vs-column-all ${EXP_LAUNCER_ROW_VS_COLUMN_STORE})

TARGET_LINK_LIBRARIES(exp-row-vs-column-all /usr/local/cuda/lib64/libcudart.so)
