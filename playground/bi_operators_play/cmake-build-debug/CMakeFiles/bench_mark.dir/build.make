# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

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
CMAKE_COMMAND = /home/pegasus/Dokumente/work/clion-2016.3.2/bin/cmake/bin/cmake

# The command to remove a file.
RM = /home/pegasus/Dokumente/work/clion-2016.3.2/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/pegasus/Dokumente/work/codes/final_develop_bi/MondrianDB/playground/bi_operators_play

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pegasus/Dokumente/work/codes/final_develop_bi/MondrianDB/playground/bi_operators_play/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/bench_mark.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/bench_mark.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/bench_mark.dir/flags.make

CMakeFiles/bench_mark.dir/bench_marking_play.cpp.o: CMakeFiles/bench_mark.dir/flags.make
CMakeFiles/bench_mark.dir/bench_marking_play.cpp.o: ../bench_marking_play.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pegasus/Dokumente/work/codes/final_develop_bi/MondrianDB/playground/bi_operators_play/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/bench_mark.dir/bench_marking_play.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/bench_mark.dir/bench_marking_play.cpp.o -c /home/pegasus/Dokumente/work/codes/final_develop_bi/MondrianDB/playground/bi_operators_play/bench_marking_play.cpp

CMakeFiles/bench_mark.dir/bench_marking_play.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bench_mark.dir/bench_marking_play.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pegasus/Dokumente/work/codes/final_develop_bi/MondrianDB/playground/bi_operators_play/bench_marking_play.cpp > CMakeFiles/bench_mark.dir/bench_marking_play.cpp.i

CMakeFiles/bench_mark.dir/bench_marking_play.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bench_mark.dir/bench_marking_play.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pegasus/Dokumente/work/codes/final_develop_bi/MondrianDB/playground/bi_operators_play/bench_marking_play.cpp -o CMakeFiles/bench_mark.dir/bench_marking_play.cpp.s

CMakeFiles/bench_mark.dir/bench_marking_play.cpp.o.requires:

.PHONY : CMakeFiles/bench_mark.dir/bench_marking_play.cpp.o.requires

CMakeFiles/bench_mark.dir/bench_marking_play.cpp.o.provides: CMakeFiles/bench_mark.dir/bench_marking_play.cpp.o.requires
	$(MAKE) -f CMakeFiles/bench_mark.dir/build.make CMakeFiles/bench_mark.dir/bench_marking_play.cpp.o.provides.build
.PHONY : CMakeFiles/bench_mark.dir/bench_marking_play.cpp.o.provides

CMakeFiles/bench_mark.dir/bench_marking_play.cpp.o.provides.build: CMakeFiles/bench_mark.dir/bench_marking_play.cpp.o


# Object files for target bench_mark
bench_mark_OBJECTS = \
"CMakeFiles/bench_mark.dir/bench_marking_play.cpp.o"

# External object files for target bench_mark
bench_mark_EXTERNAL_OBJECTS =

bench_mark: CMakeFiles/bench_mark.dir/bench_marking_play.cpp.o
bench_mark: CMakeFiles/bench_mark.dir/build.make
bench_mark: CMakeFiles/bench_mark.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pegasus/Dokumente/work/codes/final_develop_bi/MondrianDB/playground/bi_operators_play/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bench_mark"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bench_mark.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/bench_mark.dir/build: bench_mark

.PHONY : CMakeFiles/bench_mark.dir/build

CMakeFiles/bench_mark.dir/requires: CMakeFiles/bench_mark.dir/bench_marking_play.cpp.o.requires

.PHONY : CMakeFiles/bench_mark.dir/requires

CMakeFiles/bench_mark.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/bench_mark.dir/cmake_clean.cmake
.PHONY : CMakeFiles/bench_mark.dir/clean

CMakeFiles/bench_mark.dir/depend:
	cd /home/pegasus/Dokumente/work/codes/final_develop_bi/MondrianDB/playground/bi_operators_play/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pegasus/Dokumente/work/codes/final_develop_bi/MondrianDB/playground/bi_operators_play /home/pegasus/Dokumente/work/codes/final_develop_bi/MondrianDB/playground/bi_operators_play /home/pegasus/Dokumente/work/codes/final_develop_bi/MondrianDB/playground/bi_operators_play/cmake-build-debug /home/pegasus/Dokumente/work/codes/final_develop_bi/MondrianDB/playground/bi_operators_play/cmake-build-debug /home/pegasus/Dokumente/work/codes/final_develop_bi/MondrianDB/playground/bi_operators_play/cmake-build-debug/CMakeFiles/bench_mark.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/bench_mark.dir/depend

