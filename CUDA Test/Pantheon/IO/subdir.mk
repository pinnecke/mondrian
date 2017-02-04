################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CXX_SRCS += \
../Pantheon/IO/BufferManager.cxx 

OBJS += \
./Pantheon/IO/BufferManager.o 

CXX_DEPS += \
./Pantheon/IO/BufferManager.d 


# Each subdirectory must supply rules for building sources it contributes
Pantheon/IO/%.o: ../Pantheon/IO/%.cxx
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O3   -odir "Pantheon/IO" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


