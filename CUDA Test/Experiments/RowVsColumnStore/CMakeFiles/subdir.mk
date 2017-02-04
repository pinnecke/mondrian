################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Experiments/RowVsColumnStore/CMakeFiles/feature_tests.c 

CXX_SRCS += \
../Experiments/RowVsColumnStore/CMakeFiles/feature_tests.cxx 

OBJS += \
./Experiments/RowVsColumnStore/CMakeFiles/feature_tests.o 

C_DEPS += \
./Experiments/RowVsColumnStore/CMakeFiles/feature_tests.d 

CXX_DEPS += \
./Experiments/RowVsColumnStore/CMakeFiles/feature_tests.d 


# Each subdirectory must supply rules for building sources it contributes
Experiments/RowVsColumnStore/CMakeFiles/%.o: ../Experiments/RowVsColumnStore/CMakeFiles/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O3   -odir "Experiments/RowVsColumnStore/CMakeFiles" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O3 --compile  -x c -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Experiments/RowVsColumnStore/CMakeFiles/%.o: ../Experiments/RowVsColumnStore/CMakeFiles/%.cxx
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O3   -odir "Experiments/RowVsColumnStore/CMakeFiles" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


