################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Experiments/RowVsColumnStore/Shared/Common.cpp 

OBJS += \
./Experiments/RowVsColumnStore/Shared/Common.o 

CPP_DEPS += \
./Experiments/RowVsColumnStore/Shared/Common.d 


# Each subdirectory must supply rules for building sources it contributes
Experiments/RowVsColumnStore/Shared/%.o: ../Experiments/RowVsColumnStore/Shared/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O3   -odir "Experiments/RowVsColumnStore/Shared" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


