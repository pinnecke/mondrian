################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../Experiments/RowVsColumnStore/Device/Kernels/ParallelReduction.cu 

CU_DEPS += \
./Experiments/RowVsColumnStore/Device/Kernels/ParallelReduction.d 

OBJS += \
./Experiments/RowVsColumnStore/Device/Kernels/ParallelReduction.o 


# Each subdirectory must supply rules for building sources it contributes
Experiments/RowVsColumnStore/Device/Kernels/%.o: ../Experiments/RowVsColumnStore/Device/Kernels/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O3   -odir "Experiments/RowVsColumnStore/Device/Kernels" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O3 --compile --relocatable-device-code=false  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


