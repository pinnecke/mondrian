################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Experiments/RowVsColumnStore/Launcher.cpp 

OBJS += \
./Experiments/RowVsColumnStore/Launcher.o 

CPP_DEPS += \
./Experiments/RowVsColumnStore/Launcher.d 


# Each subdirectory must supply rules for building sources it contributes
Experiments/RowVsColumnStore/%.o: ../Experiments/RowVsColumnStore/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O3   -odir "Experiments/RowVsColumnStore" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


