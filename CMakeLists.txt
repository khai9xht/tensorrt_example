cmake_minimum_required(VERSION 3.10)

# set builder
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})

# set the project name
project(TensorRTInference VERSION 1.0)

# set required package
find_package(threads)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TensorRT REQUIRED)

# add the executable
add_executable(tensorrt_inference main.cpp)
target_include_directories(tensorrt_inference PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_link_libraries(tensorrt_inference PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})

