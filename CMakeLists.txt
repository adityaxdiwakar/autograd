cmake_minimum_required(VERSION 3.10)
project(autograd)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CPP_FILES
    src/val.cc)

set(HEADER_FILES
    include/val.hpp)

add_executable(autograd ${CPP_FILES} ${HEADER_FILES})
target_include_directories(autograd PRIVATE include/)
