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
include dira_pca8266_controller/CMakeFiles/controller.dir/depend.make

# Include the progress variables for this target.
include dira_pca8266_controller/CMakeFiles/controller.dir/progress.make

# Include the compile flags for this target's objects.
include dira_pca8266_controller/CMakeFiles/controller.dir/flags.make

dira_pca8266_controller/CMakeFiles/controller.dir/src/dira_pca8266_controller.cpp.o: dira_pca8266_controller/CMakeFiles/controller.dir/flags.make
dira_pca8266_controller/CMakeFiles/controller.dir/src/dira_pca8266_controller.cpp.o: /home/ubuntu/catkin_ws/src/dira_pca8266_controller/src/dira_pca8266_controller.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object dira_pca8266_controller/CMakeFiles/controller.dir/src/dira_pca8266_controller.cpp.o"
	cd /home/ubuntu/catkin_ws/build/dira_pca8266_controller && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/controller.dir/src/dira_pca8266_controller.cpp.o -c /home/ubuntu/catkin_ws/src/dira_pca8266_controller/src/dira_pca8266_controller.cpp

dira_pca8266_controller/CMakeFiles/controller.dir/src/dira_pca8266_controller.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/controller.dir/src/dira_pca8266_controller.cpp.i"
	cd /home/ubuntu/catkin_ws/build/dira_pca8266_controller && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/catkin_ws/src/dira_pca8266_controller/src/dira_pca8266_controller.cpp > CMakeFiles/controller.dir/src/dira_pca8266_controller.cpp.i

dira_pca8266_controller/CMakeFiles/controller.dir/src/dira_pca8266_controller.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/controller.dir/src/dira_pca8266_controller.cpp.s"
	cd /home/ubuntu/catkin_ws/build/dira_pca8266_controller && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/catkin_ws/src/dira_pca8266_controller/src/dira_pca8266_controller.cpp -o CMakeFiles/controller.dir/src/dira_pca8266_controller.cpp.s

dira_pca8266_controller/CMakeFiles/controller.dir/src/dira_pca8266_controller.cpp.o.requires:

.PHONY : dira_pca8266_controller/CMakeFiles/controller.dir/src/dira_pca8266_controller.cpp.o.requires

dira_pca8266_controller/CMakeFiles/controller.dir/src/dira_pca8266_controller.cpp.o.provides: dira_pca8266_controller/CMakeFiles/controller.dir/src/dira_pca8266_controller.cpp.o.requires
	$(MAKE) -f dira_pca8266_controller/CMakeFiles/controller.dir/build.make dira_pca8266_controller/CMakeFiles/controller.dir/src/dira_pca8266_controller.cpp.o.provides.build
.PHONY : dira_pca8266_controller/CMakeFiles/controller.dir/src/dira_pca8266_controller.cpp.o.provides

dira_pca8266_controller/CMakeFiles/controller.dir/src/dira_pca8266_controller.cpp.o.provides.build: dira_pca8266_controller/CMakeFiles/controller.dir/src/dira_pca8266_controller.cpp.o


# Object files for target controller
controller_OBJECTS = \
"CMakeFiles/controller.dir/src/dira_pca8266_controller.cpp.o"

# External object files for target controller
controller_EXTERNAL_OBJECTS =

/home/ubuntu/catkin_ws/devel/lib/libcontroller.so: dira_pca8266_controller/CMakeFiles/controller.dir/src/dira_pca8266_controller.cpp.o
/home/ubuntu/catkin_ws/devel/lib/libcontroller.so: dira_pca8266_controller/CMakeFiles/controller.dir/build.make
/home/ubuntu/catkin_ws/devel/lib/libcontroller.so: /home/ubuntu/catkin_ws/devel/lib/libpca9685.so
/home/ubuntu/catkin_ws/devel/lib/libcontroller.so: dira_pca8266_controller/CMakeFiles/controller.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library /home/ubuntu/catkin_ws/devel/lib/libcontroller.so"
	cd /home/ubuntu/catkin_ws/build/dira_pca8266_controller && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/controller.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
dira_pca8266_controller/CMakeFiles/controller.dir/build: /home/ubuntu/catkin_ws/devel/lib/libcontroller.so

.PHONY : dira_pca8266_controller/CMakeFiles/controller.dir/build

dira_pca8266_controller/CMakeFiles/controller.dir/requires: dira_pca8266_controller/CMakeFiles/controller.dir/src/dira_pca8266_controller.cpp.o.requires

.PHONY : dira_pca8266_controller/CMakeFiles/controller.dir/requires

dira_pca8266_controller/CMakeFiles/controller.dir/clean:
	cd /home/ubuntu/catkin_ws/build/dira_pca8266_controller && $(CMAKE_COMMAND) -P CMakeFiles/controller.dir/cmake_clean.cmake
.PHONY : dira_pca8266_controller/CMakeFiles/controller.dir/clean

dira_pca8266_controller/CMakeFiles/controller.dir/depend:
	cd /home/ubuntu/catkin_ws/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/catkin_ws/src /home/ubuntu/catkin_ws/src/dira_pca8266_controller /home/ubuntu/catkin_ws/build /home/ubuntu/catkin_ws/build/dira_pca8266_controller /home/ubuntu/catkin_ws/build/dira_pca8266_controller/CMakeFiles/controller.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : dira_pca8266_controller/CMakeFiles/controller.dir/depend

