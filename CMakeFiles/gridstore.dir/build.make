# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/marcus/cuda-workspace/Pantheon

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/marcus/cuda-workspace/Pantheon

# Include any dependencies generated for this target.
include CMakeFiles/gridstore.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/gridstore.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/gridstore.dir/flags.make

CMakeFiles/gridstore.dir/main.cxx.o: CMakeFiles/gridstore.dir/flags.make
CMakeFiles/gridstore.dir/main.cxx.o: main.cxx
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/marcus/cuda-workspace/Pantheon/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/gridstore.dir/main.cxx.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gridstore.dir/main.cxx.o -c /home/marcus/cuda-workspace/Pantheon/main.cxx

CMakeFiles/gridstore.dir/main.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gridstore.dir/main.cxx.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/marcus/cuda-workspace/Pantheon/main.cxx > CMakeFiles/gridstore.dir/main.cxx.i

CMakeFiles/gridstore.dir/main.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gridstore.dir/main.cxx.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/marcus/cuda-workspace/Pantheon/main.cxx -o CMakeFiles/gridstore.dir/main.cxx.s

CMakeFiles/gridstore.dir/main.cxx.o.requires:

.PHONY : CMakeFiles/gridstore.dir/main.cxx.o.requires

CMakeFiles/gridstore.dir/main.cxx.o.provides: CMakeFiles/gridstore.dir/main.cxx.o.requires
	$(MAKE) -f CMakeFiles/gridstore.dir/build.make CMakeFiles/gridstore.dir/main.cxx.o.provides.build
.PHONY : CMakeFiles/gridstore.dir/main.cxx.o.provides

CMakeFiles/gridstore.dir/main.cxx.o.provides.build: CMakeFiles/gridstore.dir/main.cxx.o


CMakeFiles/gridstore.dir/Pantheon/Attribute.cxx.o: CMakeFiles/gridstore.dir/flags.make
CMakeFiles/gridstore.dir/Pantheon/Attribute.cxx.o: Pantheon/Attribute.cxx
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/marcus/cuda-workspace/Pantheon/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/gridstore.dir/Pantheon/Attribute.cxx.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gridstore.dir/Pantheon/Attribute.cxx.o -c /home/marcus/cuda-workspace/Pantheon/Pantheon/Attribute.cxx

CMakeFiles/gridstore.dir/Pantheon/Attribute.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gridstore.dir/Pantheon/Attribute.cxx.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/marcus/cuda-workspace/Pantheon/Pantheon/Attribute.cxx > CMakeFiles/gridstore.dir/Pantheon/Attribute.cxx.i

CMakeFiles/gridstore.dir/Pantheon/Attribute.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gridstore.dir/Pantheon/Attribute.cxx.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/marcus/cuda-workspace/Pantheon/Pantheon/Attribute.cxx -o CMakeFiles/gridstore.dir/Pantheon/Attribute.cxx.s

CMakeFiles/gridstore.dir/Pantheon/Attribute.cxx.o.requires:

.PHONY : CMakeFiles/gridstore.dir/Pantheon/Attribute.cxx.o.requires

CMakeFiles/gridstore.dir/Pantheon/Attribute.cxx.o.provides: CMakeFiles/gridstore.dir/Pantheon/Attribute.cxx.o.requires
	$(MAKE) -f CMakeFiles/gridstore.dir/build.make CMakeFiles/gridstore.dir/Pantheon/Attribute.cxx.o.provides.build
.PHONY : CMakeFiles/gridstore.dir/Pantheon/Attribute.cxx.o.provides

CMakeFiles/gridstore.dir/Pantheon/Attribute.cxx.o.provides.build: CMakeFiles/gridstore.dir/Pantheon/Attribute.cxx.o


CMakeFiles/gridstore.dir/Pantheon/error.cxx.o: CMakeFiles/gridstore.dir/flags.make
CMakeFiles/gridstore.dir/Pantheon/error.cxx.o: Pantheon/error.cxx
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/marcus/cuda-workspace/Pantheon/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/gridstore.dir/Pantheon/error.cxx.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gridstore.dir/Pantheon/error.cxx.o -c /home/marcus/cuda-workspace/Pantheon/Pantheon/error.cxx

CMakeFiles/gridstore.dir/Pantheon/error.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gridstore.dir/Pantheon/error.cxx.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/marcus/cuda-workspace/Pantheon/Pantheon/error.cxx > CMakeFiles/gridstore.dir/Pantheon/error.cxx.i

CMakeFiles/gridstore.dir/Pantheon/error.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gridstore.dir/Pantheon/error.cxx.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/marcus/cuda-workspace/Pantheon/Pantheon/error.cxx -o CMakeFiles/gridstore.dir/Pantheon/error.cxx.s

CMakeFiles/gridstore.dir/Pantheon/error.cxx.o.requires:

.PHONY : CMakeFiles/gridstore.dir/Pantheon/error.cxx.o.requires

CMakeFiles/gridstore.dir/Pantheon/error.cxx.o.provides: CMakeFiles/gridstore.dir/Pantheon/error.cxx.o.requires
	$(MAKE) -f CMakeFiles/gridstore.dir/build.make CMakeFiles/gridstore.dir/Pantheon/error.cxx.o.provides.build
.PHONY : CMakeFiles/gridstore.dir/Pantheon/error.cxx.o.provides

CMakeFiles/gridstore.dir/Pantheon/error.cxx.o.provides.build: CMakeFiles/gridstore.dir/Pantheon/error.cxx.o


CMakeFiles/gridstore.dir/Pantheon/event.cxx.o: CMakeFiles/gridstore.dir/flags.make
CMakeFiles/gridstore.dir/Pantheon/event.cxx.o: Pantheon/event.cxx
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/marcus/cuda-workspace/Pantheon/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/gridstore.dir/Pantheon/event.cxx.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gridstore.dir/Pantheon/event.cxx.o -c /home/marcus/cuda-workspace/Pantheon/Pantheon/event.cxx

CMakeFiles/gridstore.dir/Pantheon/event.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gridstore.dir/Pantheon/event.cxx.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/marcus/cuda-workspace/Pantheon/Pantheon/event.cxx > CMakeFiles/gridstore.dir/Pantheon/event.cxx.i

CMakeFiles/gridstore.dir/Pantheon/event.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gridstore.dir/Pantheon/event.cxx.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/marcus/cuda-workspace/Pantheon/Pantheon/event.cxx -o CMakeFiles/gridstore.dir/Pantheon/event.cxx.s

CMakeFiles/gridstore.dir/Pantheon/event.cxx.o.requires:

.PHONY : CMakeFiles/gridstore.dir/Pantheon/event.cxx.o.requires

CMakeFiles/gridstore.dir/Pantheon/event.cxx.o.provides: CMakeFiles/gridstore.dir/Pantheon/event.cxx.o.requires
	$(MAKE) -f CMakeFiles/gridstore.dir/build.make CMakeFiles/gridstore.dir/Pantheon/event.cxx.o.provides.build
.PHONY : CMakeFiles/gridstore.dir/Pantheon/event.cxx.o.provides

CMakeFiles/gridstore.dir/Pantheon/event.cxx.o.provides.build: CMakeFiles/gridstore.dir/Pantheon/event.cxx.o


CMakeFiles/gridstore.dir/Pantheon/Container/Queue.cxx.o: CMakeFiles/gridstore.dir/flags.make
CMakeFiles/gridstore.dir/Pantheon/Container/Queue.cxx.o: Pantheon/Container/Queue.cxx
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/marcus/cuda-workspace/Pantheon/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/gridstore.dir/Pantheon/Container/Queue.cxx.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gridstore.dir/Pantheon/Container/Queue.cxx.o -c /home/marcus/cuda-workspace/Pantheon/Pantheon/Container/Queue.cxx

CMakeFiles/gridstore.dir/Pantheon/Container/Queue.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gridstore.dir/Pantheon/Container/Queue.cxx.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/marcus/cuda-workspace/Pantheon/Pantheon/Container/Queue.cxx > CMakeFiles/gridstore.dir/Pantheon/Container/Queue.cxx.i

CMakeFiles/gridstore.dir/Pantheon/Container/Queue.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gridstore.dir/Pantheon/Container/Queue.cxx.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/marcus/cuda-workspace/Pantheon/Pantheon/Container/Queue.cxx -o CMakeFiles/gridstore.dir/Pantheon/Container/Queue.cxx.s

CMakeFiles/gridstore.dir/Pantheon/Container/Queue.cxx.o.requires:

.PHONY : CMakeFiles/gridstore.dir/Pantheon/Container/Queue.cxx.o.requires

CMakeFiles/gridstore.dir/Pantheon/Container/Queue.cxx.o.provides: CMakeFiles/gridstore.dir/Pantheon/Container/Queue.cxx.o.requires
	$(MAKE) -f CMakeFiles/gridstore.dir/build.make CMakeFiles/gridstore.dir/Pantheon/Container/Queue.cxx.o.provides.build
.PHONY : CMakeFiles/gridstore.dir/Pantheon/Container/Queue.cxx.o.provides

CMakeFiles/gridstore.dir/Pantheon/Container/Queue.cxx.o.provides.build: CMakeFiles/gridstore.dir/Pantheon/Container/Queue.cxx.o


CMakeFiles/gridstore.dir/Pantheon/Container/RecycleBuffer.cxx.o: CMakeFiles/gridstore.dir/flags.make
CMakeFiles/gridstore.dir/Pantheon/Container/RecycleBuffer.cxx.o: Pantheon/Container/RecycleBuffer.cxx
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/marcus/cuda-workspace/Pantheon/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/gridstore.dir/Pantheon/Container/RecycleBuffer.cxx.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gridstore.dir/Pantheon/Container/RecycleBuffer.cxx.o -c /home/marcus/cuda-workspace/Pantheon/Pantheon/Container/RecycleBuffer.cxx

CMakeFiles/gridstore.dir/Pantheon/Container/RecycleBuffer.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gridstore.dir/Pantheon/Container/RecycleBuffer.cxx.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/marcus/cuda-workspace/Pantheon/Pantheon/Container/RecycleBuffer.cxx > CMakeFiles/gridstore.dir/Pantheon/Container/RecycleBuffer.cxx.i

CMakeFiles/gridstore.dir/Pantheon/Container/RecycleBuffer.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gridstore.dir/Pantheon/Container/RecycleBuffer.cxx.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/marcus/cuda-workspace/Pantheon/Pantheon/Container/RecycleBuffer.cxx -o CMakeFiles/gridstore.dir/Pantheon/Container/RecycleBuffer.cxx.s

CMakeFiles/gridstore.dir/Pantheon/Container/RecycleBuffer.cxx.o.requires:

.PHONY : CMakeFiles/gridstore.dir/Pantheon/Container/RecycleBuffer.cxx.o.requires

CMakeFiles/gridstore.dir/Pantheon/Container/RecycleBuffer.cxx.o.provides: CMakeFiles/gridstore.dir/Pantheon/Container/RecycleBuffer.cxx.o.requires
	$(MAKE) -f CMakeFiles/gridstore.dir/build.make CMakeFiles/gridstore.dir/Pantheon/Container/RecycleBuffer.cxx.o.provides.build
.PHONY : CMakeFiles/gridstore.dir/Pantheon/Container/RecycleBuffer.cxx.o.provides

CMakeFiles/gridstore.dir/Pantheon/Container/RecycleBuffer.cxx.o.provides.build: CMakeFiles/gridstore.dir/Pantheon/Container/RecycleBuffer.cxx.o


CMakeFiles/gridstore.dir/Pantheon/config.cxx.o: CMakeFiles/gridstore.dir/flags.make
CMakeFiles/gridstore.dir/Pantheon/config.cxx.o: Pantheon/config.cxx
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/marcus/cuda-workspace/Pantheon/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/gridstore.dir/Pantheon/config.cxx.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gridstore.dir/Pantheon/config.cxx.o -c /home/marcus/cuda-workspace/Pantheon/Pantheon/config.cxx

CMakeFiles/gridstore.dir/Pantheon/config.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gridstore.dir/Pantheon/config.cxx.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/marcus/cuda-workspace/Pantheon/Pantheon/config.cxx > CMakeFiles/gridstore.dir/Pantheon/config.cxx.i

CMakeFiles/gridstore.dir/Pantheon/config.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gridstore.dir/Pantheon/config.cxx.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/marcus/cuda-workspace/Pantheon/Pantheon/config.cxx -o CMakeFiles/gridstore.dir/Pantheon/config.cxx.s

CMakeFiles/gridstore.dir/Pantheon/config.cxx.o.requires:

.PHONY : CMakeFiles/gridstore.dir/Pantheon/config.cxx.o.requires

CMakeFiles/gridstore.dir/Pantheon/config.cxx.o.provides: CMakeFiles/gridstore.dir/Pantheon/config.cxx.o.requires
	$(MAKE) -f CMakeFiles/gridstore.dir/build.make CMakeFiles/gridstore.dir/Pantheon/config.cxx.o.provides.build
.PHONY : CMakeFiles/gridstore.dir/Pantheon/config.cxx.o.provides

CMakeFiles/gridstore.dir/Pantheon/config.cxx.o.provides.build: CMakeFiles/gridstore.dir/Pantheon/config.cxx.o


CMakeFiles/gridstore.dir/Experiments/PagingVsBuffering/launcher.cpp.o: CMakeFiles/gridstore.dir/flags.make
CMakeFiles/gridstore.dir/Experiments/PagingVsBuffering/launcher.cpp.o: Experiments/PagingVsBuffering/launcher.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/marcus/cuda-workspace/Pantheon/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/gridstore.dir/Experiments/PagingVsBuffering/launcher.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gridstore.dir/Experiments/PagingVsBuffering/launcher.cpp.o -c /home/marcus/cuda-workspace/Pantheon/Experiments/PagingVsBuffering/launcher.cpp

CMakeFiles/gridstore.dir/Experiments/PagingVsBuffering/launcher.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gridstore.dir/Experiments/PagingVsBuffering/launcher.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/marcus/cuda-workspace/Pantheon/Experiments/PagingVsBuffering/launcher.cpp > CMakeFiles/gridstore.dir/Experiments/PagingVsBuffering/launcher.cpp.i

CMakeFiles/gridstore.dir/Experiments/PagingVsBuffering/launcher.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gridstore.dir/Experiments/PagingVsBuffering/launcher.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/marcus/cuda-workspace/Pantheon/Experiments/PagingVsBuffering/launcher.cpp -o CMakeFiles/gridstore.dir/Experiments/PagingVsBuffering/launcher.cpp.s

CMakeFiles/gridstore.dir/Experiments/PagingVsBuffering/launcher.cpp.o.requires:

.PHONY : CMakeFiles/gridstore.dir/Experiments/PagingVsBuffering/launcher.cpp.o.requires

CMakeFiles/gridstore.dir/Experiments/PagingVsBuffering/launcher.cpp.o.provides: CMakeFiles/gridstore.dir/Experiments/PagingVsBuffering/launcher.cpp.o.requires
	$(MAKE) -f CMakeFiles/gridstore.dir/build.make CMakeFiles/gridstore.dir/Experiments/PagingVsBuffering/launcher.cpp.o.provides.build
.PHONY : CMakeFiles/gridstore.dir/Experiments/PagingVsBuffering/launcher.cpp.o.provides

CMakeFiles/gridstore.dir/Experiments/PagingVsBuffering/launcher.cpp.o.provides.build: CMakeFiles/gridstore.dir/Experiments/PagingVsBuffering/launcher.cpp.o


CMakeFiles/gridstore.dir/Pantheon/IO/BufferManager.cxx.o: CMakeFiles/gridstore.dir/flags.make
CMakeFiles/gridstore.dir/Pantheon/IO/BufferManager.cxx.o: Pantheon/IO/BufferManager.cxx
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/marcus/cuda-workspace/Pantheon/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/gridstore.dir/Pantheon/IO/BufferManager.cxx.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gridstore.dir/Pantheon/IO/BufferManager.cxx.o -c /home/marcus/cuda-workspace/Pantheon/Pantheon/IO/BufferManager.cxx

CMakeFiles/gridstore.dir/Pantheon/IO/BufferManager.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gridstore.dir/Pantheon/IO/BufferManager.cxx.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/marcus/cuda-workspace/Pantheon/Pantheon/IO/BufferManager.cxx > CMakeFiles/gridstore.dir/Pantheon/IO/BufferManager.cxx.i

CMakeFiles/gridstore.dir/Pantheon/IO/BufferManager.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gridstore.dir/Pantheon/IO/BufferManager.cxx.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/marcus/cuda-workspace/Pantheon/Pantheon/IO/BufferManager.cxx -o CMakeFiles/gridstore.dir/Pantheon/IO/BufferManager.cxx.s

CMakeFiles/gridstore.dir/Pantheon/IO/BufferManager.cxx.o.requires:

.PHONY : CMakeFiles/gridstore.dir/Pantheon/IO/BufferManager.cxx.o.requires

CMakeFiles/gridstore.dir/Pantheon/IO/BufferManager.cxx.o.provides: CMakeFiles/gridstore.dir/Pantheon/IO/BufferManager.cxx.o.requires
	$(MAKE) -f CMakeFiles/gridstore.dir/build.make CMakeFiles/gridstore.dir/Pantheon/IO/BufferManager.cxx.o.provides.build
.PHONY : CMakeFiles/gridstore.dir/Pantheon/IO/BufferManager.cxx.o.provides

CMakeFiles/gridstore.dir/Pantheon/IO/BufferManager.cxx.o.provides.build: CMakeFiles/gridstore.dir/Pantheon/IO/BufferManager.cxx.o


CMakeFiles/gridstore.dir/Pantheon/Container/ArrayList.cxx.o: CMakeFiles/gridstore.dir/flags.make
CMakeFiles/gridstore.dir/Pantheon/Container/ArrayList.cxx.o: Pantheon/Container/ArrayList.cxx
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/marcus/cuda-workspace/Pantheon/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/gridstore.dir/Pantheon/Container/ArrayList.cxx.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gridstore.dir/Pantheon/Container/ArrayList.cxx.o -c /home/marcus/cuda-workspace/Pantheon/Pantheon/Container/ArrayList.cxx

CMakeFiles/gridstore.dir/Pantheon/Container/ArrayList.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gridstore.dir/Pantheon/Container/ArrayList.cxx.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/marcus/cuda-workspace/Pantheon/Pantheon/Container/ArrayList.cxx > CMakeFiles/gridstore.dir/Pantheon/Container/ArrayList.cxx.i

CMakeFiles/gridstore.dir/Pantheon/Container/ArrayList.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gridstore.dir/Pantheon/Container/ArrayList.cxx.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/marcus/cuda-workspace/Pantheon/Pantheon/Container/ArrayList.cxx -o CMakeFiles/gridstore.dir/Pantheon/Container/ArrayList.cxx.s

CMakeFiles/gridstore.dir/Pantheon/Container/ArrayList.cxx.o.requires:

.PHONY : CMakeFiles/gridstore.dir/Pantheon/Container/ArrayList.cxx.o.requires

CMakeFiles/gridstore.dir/Pantheon/Container/ArrayList.cxx.o.provides: CMakeFiles/gridstore.dir/Pantheon/Container/ArrayList.cxx.o.requires
	$(MAKE) -f CMakeFiles/gridstore.dir/build.make CMakeFiles/gridstore.dir/Pantheon/Container/ArrayList.cxx.o.provides.build
.PHONY : CMakeFiles/gridstore.dir/Pantheon/Container/ArrayList.cxx.o.provides

CMakeFiles/gridstore.dir/Pantheon/Container/ArrayList.cxx.o.provides.build: CMakeFiles/gridstore.dir/Pantheon/Container/ArrayList.cxx.o


CMakeFiles/gridstore.dir/Pantheon/Container/LinkedList.cxx.o: CMakeFiles/gridstore.dir/flags.make
CMakeFiles/gridstore.dir/Pantheon/Container/LinkedList.cxx.o: Pantheon/Container/LinkedList.cxx
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/marcus/cuda-workspace/Pantheon/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object CMakeFiles/gridstore.dir/Pantheon/Container/LinkedList.cxx.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gridstore.dir/Pantheon/Container/LinkedList.cxx.o -c /home/marcus/cuda-workspace/Pantheon/Pantheon/Container/LinkedList.cxx

CMakeFiles/gridstore.dir/Pantheon/Container/LinkedList.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gridstore.dir/Pantheon/Container/LinkedList.cxx.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/marcus/cuda-workspace/Pantheon/Pantheon/Container/LinkedList.cxx > CMakeFiles/gridstore.dir/Pantheon/Container/LinkedList.cxx.i

CMakeFiles/gridstore.dir/Pantheon/Container/LinkedList.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gridstore.dir/Pantheon/Container/LinkedList.cxx.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/marcus/cuda-workspace/Pantheon/Pantheon/Container/LinkedList.cxx -o CMakeFiles/gridstore.dir/Pantheon/Container/LinkedList.cxx.s

CMakeFiles/gridstore.dir/Pantheon/Container/LinkedList.cxx.o.requires:

.PHONY : CMakeFiles/gridstore.dir/Pantheon/Container/LinkedList.cxx.o.requires

CMakeFiles/gridstore.dir/Pantheon/Container/LinkedList.cxx.o.provides: CMakeFiles/gridstore.dir/Pantheon/Container/LinkedList.cxx.o.requires
	$(MAKE) -f CMakeFiles/gridstore.dir/build.make CMakeFiles/gridstore.dir/Pantheon/Container/LinkedList.cxx.o.provides.build
.PHONY : CMakeFiles/gridstore.dir/Pantheon/Container/LinkedList.cxx.o.provides

CMakeFiles/gridstore.dir/Pantheon/Container/LinkedList.cxx.o.provides.build: CMakeFiles/gridstore.dir/Pantheon/Container/LinkedList.cxx.o


# Object files for target gridstore
gridstore_OBJECTS = \
"CMakeFiles/gridstore.dir/main.cxx.o" \
"CMakeFiles/gridstore.dir/Pantheon/Attribute.cxx.o" \
"CMakeFiles/gridstore.dir/Pantheon/error.cxx.o" \
"CMakeFiles/gridstore.dir/Pantheon/event.cxx.o" \
"CMakeFiles/gridstore.dir/Pantheon/Container/Queue.cxx.o" \
"CMakeFiles/gridstore.dir/Pantheon/Container/RecycleBuffer.cxx.o" \
"CMakeFiles/gridstore.dir/Pantheon/config.cxx.o" \
"CMakeFiles/gridstore.dir/Experiments/PagingVsBuffering/launcher.cpp.o" \
"CMakeFiles/gridstore.dir/Pantheon/IO/BufferManager.cxx.o" \
"CMakeFiles/gridstore.dir/Pantheon/Container/ArrayList.cxx.o" \
"CMakeFiles/gridstore.dir/Pantheon/Container/LinkedList.cxx.o"

# External object files for target gridstore
gridstore_EXTERNAL_OBJECTS =

gridstore: CMakeFiles/gridstore.dir/main.cxx.o
gridstore: CMakeFiles/gridstore.dir/Pantheon/Attribute.cxx.o
gridstore: CMakeFiles/gridstore.dir/Pantheon/error.cxx.o
gridstore: CMakeFiles/gridstore.dir/Pantheon/event.cxx.o
gridstore: CMakeFiles/gridstore.dir/Pantheon/Container/Queue.cxx.o
gridstore: CMakeFiles/gridstore.dir/Pantheon/Container/RecycleBuffer.cxx.o
gridstore: CMakeFiles/gridstore.dir/Pantheon/config.cxx.o
gridstore: CMakeFiles/gridstore.dir/Experiments/PagingVsBuffering/launcher.cpp.o
gridstore: CMakeFiles/gridstore.dir/Pantheon/IO/BufferManager.cxx.o
gridstore: CMakeFiles/gridstore.dir/Pantheon/Container/ArrayList.cxx.o
gridstore: CMakeFiles/gridstore.dir/Pantheon/Container/LinkedList.cxx.o
gridstore: CMakeFiles/gridstore.dir/build.make
gridstore: CMakeFiles/gridstore.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/marcus/cuda-workspace/Pantheon/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Linking CXX executable gridstore"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gridstore.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/gridstore.dir/build: gridstore

.PHONY : CMakeFiles/gridstore.dir/build

CMakeFiles/gridstore.dir/requires: CMakeFiles/gridstore.dir/main.cxx.o.requires
CMakeFiles/gridstore.dir/requires: CMakeFiles/gridstore.dir/Pantheon/Attribute.cxx.o.requires
CMakeFiles/gridstore.dir/requires: CMakeFiles/gridstore.dir/Pantheon/error.cxx.o.requires
CMakeFiles/gridstore.dir/requires: CMakeFiles/gridstore.dir/Pantheon/event.cxx.o.requires
CMakeFiles/gridstore.dir/requires: CMakeFiles/gridstore.dir/Pantheon/Container/Queue.cxx.o.requires
CMakeFiles/gridstore.dir/requires: CMakeFiles/gridstore.dir/Pantheon/Container/RecycleBuffer.cxx.o.requires
CMakeFiles/gridstore.dir/requires: CMakeFiles/gridstore.dir/Pantheon/config.cxx.o.requires
CMakeFiles/gridstore.dir/requires: CMakeFiles/gridstore.dir/Experiments/PagingVsBuffering/launcher.cpp.o.requires
CMakeFiles/gridstore.dir/requires: CMakeFiles/gridstore.dir/Pantheon/IO/BufferManager.cxx.o.requires
CMakeFiles/gridstore.dir/requires: CMakeFiles/gridstore.dir/Pantheon/Container/ArrayList.cxx.o.requires
CMakeFiles/gridstore.dir/requires: CMakeFiles/gridstore.dir/Pantheon/Container/LinkedList.cxx.o.requires

.PHONY : CMakeFiles/gridstore.dir/requires

CMakeFiles/gridstore.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/gridstore.dir/cmake_clean.cmake
.PHONY : CMakeFiles/gridstore.dir/clean

CMakeFiles/gridstore.dir/depend:
	cd /home/marcus/cuda-workspace/Pantheon && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/marcus/cuda-workspace/Pantheon /home/marcus/cuda-workspace/Pantheon /home/marcus/cuda-workspace/Pantheon /home/marcus/cuda-workspace/Pantheon /home/marcus/cuda-workspace/Pantheon/CMakeFiles/gridstore.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/gridstore.dir/depend
