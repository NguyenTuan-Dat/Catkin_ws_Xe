# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_SOURCE_DIR = /home/ubuntu/catkin_ws/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/catkin_ws/build

# Include any dependencies generated for this target.
include hal_ros/CMakeFiles/hal.dir/depend.make

# Include the progress variables for this target.
include hal_ros/CMakeFiles/hal.dir/progress.make

# Include the compile flags for this target's objects.
include hal_ros/CMakeFiles/hal.dir/flags.make

hal_ros/CMakeFiles/hal.dir/src/api_hal.cpp.o: hal_ros/CMakeFiles/hal.dir/flags.make
hal_ros/CMakeFiles/hal.dir/src/api_hal.cpp.o: /home/ubuntu/catkin_ws/src/hal_ros/src/api_hal.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object hal_ros/CMakeFiles/hal.dir/src/api_hal.cpp.o"
	cd /home/ubuntu/catkin_ws/build/hal_ros && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hal.dir/src/api_hal.cpp.o -c /home/ubuntu/catkin_ws/src/hal_ros/src/api_hal.cpp

hal_ros/CMakeFiles/hal.dir/src/api_hal.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hal.dir/src/api_hal.cpp.i"
	cd /home/ubuntu/catkin_ws/build/hal_ros && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/catkin_ws/src/hal_ros/src/api_hal.cpp > CMakeFiles/hal.dir/src/api_hal.cpp.i

hal_ros/CMakeFiles/hal.dir/src/api_hal.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hal.dir/src/api_hal.cpp.s"
	cd /home/ubuntu/catkin_ws/build/hal_ros && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/catkin_ws/src/hal_ros/src/api_hal.cpp -o CMakeFiles/hal.dir/src/api_hal.cpp.s

hal_ros/CMakeFiles/hal.dir/src/api_hal.cpp.o.requires:

.PHONY : hal_ros/CMakeFiles/hal.dir/src/api_hal.cpp.o.requires

hal_ros/CMakeFiles/hal.dir/src/api_hal.cpp.o.provides: hal_ros/CMakeFiles/hal.dir/src/api_hal.cpp.o.requires
	$(MAKE) -f hal_ros/CMakeFiles/hal.dir/build.make hal_ros/CMakeFiles/hal.dir/src/api_hal.cpp.o.provides.build
.PHONY : hal_ros/CMakeFiles/hal.dir/src/api_hal.cpp.o.provides

hal_ros/CMakeFiles/hal.dir/src/api_hal.cpp.o.provides.build: hal_ros/CMakeFiles/hal.dir/src/api_hal.cpp.o


hal_ros/CMakeFiles/hal.dir/src/Hal.cpp.o: hal_ros/CMakeFiles/hal.dir/flags.make
hal_ros/CMakeFiles/hal.dir/src/Hal.cpp.o: /home/ubuntu/catkin_ws/src/hal_ros/src/Hal.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object hal_ros/CMakeFiles/hal.dir/src/Hal.cpp.o"
	cd /home/ubuntu/catkin_ws/build/hal_ros && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hal.dir/src/Hal.cpp.o -c /home/ubuntu/catkin_ws/src/hal_ros/src/Hal.cpp

hal_ros/CMakeFiles/hal.dir/src/Hal.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hal.dir/src/Hal.cpp.i"
	cd /home/ubuntu/catkin_ws/build/hal_ros && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/catkin_ws/src/hal_ros/src/Hal.cpp > CMakeFiles/hal.dir/src/Hal.cpp.i

hal_ros/CMakeFiles/hal.dir/src/Hal.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hal.dir/src/Hal.cpp.s"
	cd /home/ubuntu/catkin_ws/build/hal_ros && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/catkin_ws/src/hal_ros/src/Hal.cpp -o CMakeFiles/hal.dir/src/Hal.cpp.s

hal_ros/CMakeFiles/hal.dir/src/Hal.cpp.o.requires:

.PHONY : hal_ros/CMakeFiles/hal.dir/src/Hal.cpp.o.requires

hal_ros/CMakeFiles/hal.dir/src/Hal.cpp.o.provides: hal_ros/CMakeFiles/hal.dir/src/Hal.cpp.o.requires
	$(MAKE) -f hal_ros/CMakeFiles/hal.dir/build.make hal_ros/CMakeFiles/hal.dir/src/Hal.cpp.o.provides.build
.PHONY : hal_ros/CMakeFiles/hal.dir/src/Hal.cpp.o.provides

hal_ros/CMakeFiles/hal.dir/src/Hal.cpp.o.provides.build: hal_ros/CMakeFiles/hal.dir/src/Hal.cpp.o


hal_ros/CMakeFiles/hal.dir/src/LCDI2C.cpp.o: hal_ros/CMakeFiles/hal.dir/flags.make
hal_ros/CMakeFiles/hal.dir/src/LCDI2C.cpp.o: /home/ubuntu/catkin_ws/src/hal_ros/src/LCDI2C.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object hal_ros/CMakeFiles/hal.dir/src/LCDI2C.cpp.o"
	cd /home/ubuntu/catkin_ws/build/hal_ros && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hal.dir/src/LCDI2C.cpp.o -c /home/ubuntu/catkin_ws/src/hal_ros/src/LCDI2C.cpp

hal_ros/CMakeFiles/hal.dir/src/LCDI2C.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hal.dir/src/LCDI2C.cpp.i"
	cd /home/ubuntu/catkin_ws/build/hal_ros && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/catkin_ws/src/hal_ros/src/LCDI2C.cpp > CMakeFiles/hal.dir/src/LCDI2C.cpp.i

hal_ros/CMakeFiles/hal.dir/src/LCDI2C.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hal.dir/src/LCDI2C.cpp.s"
	cd /home/ubuntu/catkin_ws/build/hal_ros && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/catkin_ws/src/hal_ros/src/LCDI2C.cpp -o CMakeFiles/hal.dir/src/LCDI2C.cpp.s

hal_ros/CMakeFiles/hal.dir/src/LCDI2C.cpp.o.requires:

.PHONY : hal_ros/CMakeFiles/hal.dir/src/LCDI2C.cpp.o.requires

hal_ros/CMakeFiles/hal.dir/src/LCDI2C.cpp.o.provides: hal_ros/CMakeFiles/hal.dir/src/LCDI2C.cpp.o.requires
	$(MAKE) -f hal_ros/CMakeFiles/hal.dir/build.make hal_ros/CMakeFiles/hal.dir/src/LCDI2C.cpp.o.provides.build
.PHONY : hal_ros/CMakeFiles/hal.dir/src/LCDI2C.cpp.o.provides

hal_ros/CMakeFiles/hal.dir/src/LCDI2C.cpp.o.provides.build: hal_ros/CMakeFiles/hal.dir/src/LCDI2C.cpp.o


# Object files for target hal
hal_OBJECTS = \
"CMakeFiles/hal.dir/src/api_hal.cpp.o" \
"CMakeFiles/hal.dir/src/Hal.cpp.o" \
"CMakeFiles/hal.dir/src/LCDI2C.cpp.o"

# External object files for target hal
hal_EXTERNAL_OBJECTS =

/home/ubuntu/catkin_ws/devel/lib/libhal.so: hal_ros/CMakeFiles/hal.dir/src/api_hal.cpp.o
/home/ubuntu/catkin_ws/devel/lib/libhal.so: hal_ros/CMakeFiles/hal.dir/src/Hal.cpp.o
/home/ubuntu/catkin_ws/devel/lib/libhal.so: hal_ros/CMakeFiles/hal.dir/src/LCDI2C.cpp.o
/home/ubuntu/catkin_ws/devel/lib/libhal.so: hal_ros/CMakeFiles/hal.dir/build.make
/home/ubuntu/catkin_ws/devel/lib/libhal.so: hal_ros/CMakeFiles/hal.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX shared library /home/ubuntu/catkin_ws/devel/lib/libhal.so"
	cd /home/ubuntu/catkin_ws/build/hal_ros && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hal.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
hal_ros/CMakeFiles/hal.dir/build: /home/ubuntu/catkin_ws/devel/lib/libhal.so

.PHONY : hal_ros/CMakeFiles/hal.dir/build

hal_ros/CMakeFiles/hal.dir/requires: hal_ros/CMakeFiles/hal.dir/src/api_hal.cpp.o.requires
hal_ros/CMakeFiles/hal.dir/requires: hal_ros/CMakeFiles/hal.dir/src/Hal.cpp.o.requires
hal_ros/CMakeFiles/hal.dir/requires: hal_ros/CMakeFiles/hal.dir/src/LCDI2C.cpp.o.requires

.PHONY : hal_ros/CMakeFiles/hal.dir/requires

hal_ros/CMakeFiles/hal.dir/clean:
	cd /home/ubuntu/catkin_ws/build/hal_ros && $(CMAKE_COMMAND) -P CMakeFiles/hal.dir/cmake_clean.cmake
.PHONY : hal_ros/CMakeFiles/hal.dir/clean

hal_ros/CMakeFiles/hal.dir/depend:
	cd /home/ubuntu/catkin_ws/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/catkin_ws/src /home/ubuntu/catkin_ws/src/hal_ros /home/ubuntu/catkin_ws/build /home/ubuntu/catkin_ws/build/hal_ros /home/ubuntu/catkin_ws/build/hal_ros/CMakeFiles/hal.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : hal_ros/CMakeFiles/hal.dir/depend
