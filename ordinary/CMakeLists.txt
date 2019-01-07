cmake_minimum_required(VERSION 2.8)

project(gaussian_ordinary)

set(EXECUTABLE_OUTPUT_PATH "..")
set(CMAKE_BUILD TYPE Debug)

# OpenCV stuff
#find_package(OpenCV REQUIRED)

add_executable(gaussian_ordinary main.cpp)

#target_link_libraries(haar2d ${OpenCV_LIBS})

