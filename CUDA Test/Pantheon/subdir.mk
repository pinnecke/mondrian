################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CXX_SRCS += \
../Pantheon/Attribute.cxx \
../Pantheon/config.cxx \
../Pantheon/error.cxx \
../Pantheon/event.cxx 

OBJS += \
./Pantheon/Attribute.o \
./Pantheon/config.o \
./Pantheon/error.o \
./Pantheon/event.o 

CXX_DEPS += \
./Pantheon/Attribute.d \
./Pantheon/config.d \
./Pantheon/error.d \
./Pantheon/event.d 


# Each subdirectory must supply rules for building sources it contributes
Pantheon/%.o: ../Pantheon/%.cxx
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O3   -odir "Pantheon" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


