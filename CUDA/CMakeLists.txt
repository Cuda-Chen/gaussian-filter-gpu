cmake_minimum_required(VERSION 2.8)

project(gaussian_cuda)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
#set(CUDA_64_BIT_DEVICE_CODE ON)
#set(CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_BUILD_TYPE Debug)
set(EXECUTABLE_OUTPUT_PATH "..")
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}	-gencode arch=compute_61,code=sm_61)

cuda_add_executable(gaussian_cuda main.cpp gaussian.hpp gaussian.cu)

target_link_libraries(gaussian_cuda ${OpenCV_LIBS})
