cmake_minimum_required(VERSION 3.18)
project(rvm-vcam LANGUAGES CXX CUDA)

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA
find_package(CUDAToolkit REQUIRED)

# TensorRT (multiarch path on Ubuntu)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/x86_64-linux-gnu /usr/include
    REQUIRED
)
find_library(NVINFER_LIB nvinfer REQUIRED)
find_library(NVONNXPARSER_LIB nvonnxparser REQUIRED)

# nvJPEG for GPU-accelerated MJPEG decode
find_library(NVJPEG_LIB nvjpeg REQUIRED)

# Executable
add_executable(rvm-vcam
    src/main.cpp
    src/pipeline.cpp
    src/v4l2_capture.cpp
    src/v4l2_output.cpp
    src/trt_engine.cpp
    src/cuda_kernels.cu
)

target_include_directories(rvm-vcam PRIVATE
    ${TENSORRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    src/
)

target_link_libraries(rvm-vcam PRIVATE
    ${NVINFER_LIB}
    ${NVONNXPARSER_LIB}
    ${NVJPEG_LIB}
    CUDA::cudart
)

# Generate compile_commands.json for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
