cmake_minimum_required(VERSION 2.8)

include_directories(
		include
)

project(
	row-vs-column-store
)

include_directories(
		include
)

set(
	CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_50,code=sm_50;--std=c++11;-O3
)

set(
	CUDA_HOST_COMPILER clang++
)

find_package(
	CUDA REQUIRED
)

include_directories(
	${CUDA_INCLUDE_DIRS}
)

CUDA_ADD_LIBRARY(
	cuda_obj STATIC Device/Query.cu
)

set(
	CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -Wall"
)

add_executable(
		main
		Host/Query.cpp
		Shared/Common.cpp
		Launcher.cpp
)

target_link_libraries(
		main ${CUDA_LIBRARIES}
)

target_link_libraries(
		main cuda_obj
)

install(TARGETS main 
	RUNTIME DESTINATION bin
)