################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include subdir.mk
-include Pantheon/subdir.mk
-include Pantheon/IO/subdir.mk
-include Pantheon/Container/subdir.mk
-include Experiments/RowVsColumnStore/Shared/subdir.mk
-include Experiments/RowVsColumnStore/subdir.mk
-include Experiments/RowVsColumnStore/Host/subdir.mk
-include Experiments/RowVsColumnStore/Device/subdir.mk
-include Experiments/RowVsColumnStore/Device/Kernels/subdir.mk
-include Experiments/RowVsColumnStore/CMakeFiles/main.dir/Shared/subdir.mk
-include Experiments/RowVsColumnStore/CMakeFiles/main.dir/subdir.mk
-include Experiments/RowVsColumnStore/CMakeFiles/main.dir/Host/subdir.mk
-include Experiments/RowVsColumnStore/CMakeFiles/subdir.mk
-include Experiments/RowVsColumnStore/CMakeFiles/cuda_obj.dir/Device/subdir.mk
-include Experiments/RowVsColumnStore/CMakeFiles/3.5.1/CompilerIdCXX/subdir.mk
-include Experiments/RowVsColumnStore/CMakeFiles/3.5.1/CompilerIdC/subdir.mk
-include Experiments/PagingVsBuffering/subdir.mk
-include CMakeFiles/lbmslv.dir/Experiments/RowVsColumnStore/Shared/subdir.mk
-include CMakeFiles/lbmslv.dir/Experiments/RowVsColumnStore/subdir.mk
-include CMakeFiles/lbmslv.dir/Experiments/RowVsColumnStore/Host/subdir.mk
-include CMakeFiles/gridstore.dir/subdir.mk
-include CMakeFiles/gridstore.dir/Pantheon/subdir.mk
-include CMakeFiles/subdir.mk
-include CMakeFiles/exp_row_vs_column_store.dir/Experiments/RowVsColumnStore/Shared/subdir.mk
-include CMakeFiles/exp_row_vs_column_store.dir/Experiments/RowVsColumnStore/subdir.mk
-include CMakeFiles/exp_row_vs_column_store.dir/Experiments/RowVsColumnStore/Host/subdir.mk
-include CMakeFiles/exp-row-vs-column-all.dir/Experiments/RowVsColumnStore/Shared/subdir.mk
-include CMakeFiles/exp-row-vs-column-all.dir/Experiments/RowVsColumnStore/Host/subdir.mk
-include CMakeFiles/3.5.1/CompilerIdCXX/subdir.mk
-include CMakeFiles/3.5.1/CompilerIdC/subdir.mk

OS_SUFFIX := $(subst Linux,linux,$(subst Darwin/x86_64,darwin,$(shell uname -s)/$(shell uname -m)))

-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(CU_DEPS)),)
-include $(CU_DEPS)
endif
ifneq ($(strip $(C++_DEPS)),)
-include $(C++_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
ifneq ($(strip $(CC_DEPS)),)
-include $(CC_DEPS)
endif
ifneq ($(strip $(CPP_DEPS)),)
-include $(CPP_DEPS)
endif
ifneq ($(strip $(CXX_DEPS)),)
-include $(CXX_DEPS)
endif
ifneq ($(strip $(C_UPPER_DEPS)),)
-include $(C_UPPER_DEPS)
endif
endif

-include ../makefile.defs

# OpenGL+GLUT OS-specific define
ifeq ($(shell uname -s),Darwin)
GLUT_LIBS := -Xlinker -framework -Xlinker GLUT -Xlinker -framework -Xlinker OpenGL
else
GLUT_LIBS := -lGL -lGLU -lglut 
endif


# Add inputs and outputs from these tool invocations to the build variables 

# All Target
all: Pantheon

# Tool invocations
Pantheon: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: NVCC Linker'
	/usr/local/cuda-8.0/bin/nvcc --cudart static --relocatable-device-code=false -link -o  "Pantheon" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(CU_DEPS)$(OBJS)$(C++_DEPS)$(C_DEPS)$(CC_DEPS)$(CPP_DEPS)$(EXECUTABLES)$(CXX_DEPS)$(C_UPPER_DEPS) Pantheon
	-@echo ' '

.PHONY: all clean dependents
.SECONDARY:

-include ../makefile.targets
