# Computer Graphics, Littau Project, PBR


Build instructions for Linux
============================

Dependencies
------------
gcc/g++, CMake 2.8 (or higher), make, OpenGL, Assimp

These dependencies can be installed via the package manager of your
Linux distribution.

CMake might complain if the additional libraries Xi, Xmu, Xrandr, and Xcursor
are not installed. On, e.g., Ubuntu 14.04, you can install these libraries with
the command

  sudo apt-get install libxmu-dev libxi-dev libxrandr-dev libxcursor-dev

GLEW and GLFW will be built on-the-fly when you build the assignment
program.

Environment variables
---------------------
The environment variable ASSIGNMENT1_ROOT must be set pointing to the
extracted assignment1 directory. Example:

  export LITTAU_ROOT=$HOME/Littau

Building and running the program
-------------------------------------------
To build the program, open a terminal, navigate to
$LITTAU_ROOT/model_viewer and type

  mkdir build
  cd build
  cmake ../
  make

To run the program, navigate to the resulting executable (part1), and
type

  ./model_viewer

Alternatively, run the attached script build.sh (which will perform
all these steps for you):

  ./build.sh

Note: You don't have to run CMake every time you change something in
the source files. Just use the generated makefile (or the build.sh
script) to rebuild the program.
