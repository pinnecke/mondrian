################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../Experiments/RowVsColumnStore/Device/Query.cu 

CU_DEPS += \
./Experiments/RowVsColumnStore/Device/Query.d 

OBJS += \
./Experiments/RowVsColumnStore/Device/Query.o 


# Each subdirectory must supply rules for building sources it contributes
Experiments/RowVsColumnStore/Device/%.o: ../Experiments/RowVsColumnStore/Device/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O3   -odir "Experiments/RowVsColumnStore/Device" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O3 --compile --relocatable-device-code=false  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


