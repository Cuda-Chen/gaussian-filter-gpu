cmake_minimum_required(VERSION 3.1)

project(gaussian_cl)

set(EXECUTABLE_OUTPUT_PATH "..")
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

add_executable(gaussian_cl main.cpp gaussian_cl.cpp gaussian_cl.hpp)

target_include_directories(gaussian_cl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gaussian_cl ${OpenCV_LIBS} ${OpenCL_LIBRARY})
