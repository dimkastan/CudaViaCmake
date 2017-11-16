# cmake ..
# or
# cmake .. -DCMAKE_BUILD_TYPE=RELEASE
# Build 
# msbuild main.vcxproj
# msbuild main.vcxproj /property:Configuration=Release
#
#

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(CudaGettingStarted)

if(MSVC)
message("MSVC selected")
endif()
# find all packages needed
FIND_PACKAGE(CUDA REQUIRED)
INCLUDE(FindCUDA)
INCLUDE(cuDNN.cmake)
 
SET(BUILD_SHARED_LIBS ON)
 
SET(CMAKE_BUILD_TYPE CMAKE_CXX_FLAGS_RELEASE)

set(CUDA_VERBOSE_BUILD ON)

# compile cuda files
file(GLOB CUDA_FILES cufiles/device.cu)
list(APPEND CUDA_NVCC_FLAGS -Xcompiler) 
list(APPEND CUDA_NVCC_FLAGS --compiler-options -use_fast_math -std=c++11)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_53,code=sm_53) 
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60) 
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_62,code=sm_62) 
CUDA_COMPILE(cugen ${CUDA_FILES})

 
set_source_files_properties( ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp  PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ )

# compiple c++ code and link with cuda 
cuda_add_executable(main ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp ${cugen})
target_link_libraries(main ${CUDNN_LIBRARY} ${CUDA_CUDART_LIBRARY}  ${CUDA_curand_LIBRARY} )

list(APPEND CMAKE_CXX_FLAGS "-O3 -ffast-math -Wall")

 
