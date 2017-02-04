################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Experiments/PagingVsBuffering/launcher.cpp 

OBJS += \
./Experiments/PagingVsBuffering/launcher.o 

CPP_DEPS += \
./Experiments/PagingVsBuffering/launcher.d 


# Each subdirectory must supply rules for building sources it contributes
Experiments/PagingVsBuffering/%.o: ../Experiments/PagingVsBuffering/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O3   -odir "Experiments/PagingVsBuffering" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


