################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Experiments/RowVsColumnStore/Host/Query.cpp 

OBJS += \
./Experiments/RowVsColumnStore/Host/Query.o 

CPP_DEPS += \
./Experiments/RowVsColumnStore/Host/Query.d 


# Each subdirectory must supply rules for building sources it contributes
Experiments/RowVsColumnStore/Host/%.o: ../Experiments/RowVsColumnStore/Host/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O3   -odir "Experiments/RowVsColumnStore/Host" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


