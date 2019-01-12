cmake_minimum_required(VERSION 2.8)

project(gaussian_ordinary)

set(EXECUTABLE_OUTPUT_PATH "..")
set(CMAKE_BUILD_TYPE Debug)

# OpenCV stuff
find_package(OpenCV REQUIRED)

add_executable(gaussian_ordinary main.cpp gaussian.hpp gaussian.cpp)

target_link_libraries(gaussian_ordinary ${OpenCV_LIBS})

