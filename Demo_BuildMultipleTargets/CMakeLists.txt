# cmake ..
# or
# cmake .. -DCMAKE_BUILD_TYPE=RELEASE
# Build 
# msbuild main.vcxproj
# msbuild main.vcxproj /property:Configuration=Release
#
#

CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(CudaGettingStarted)

if(MSVC)
message("MSVC selected")
endif()
# find all packages needed
FIND_PACKAGE(CUDA REQUIRED)
INCLUDE(FindCUDA)
 
SET(BUILD_SHARED_LIBS ON)
 
SET(CMAKE_BUILD_TYPE CMAKE_CXX_FLAGS_RELEASE)

set(CUDA_VERBOSE_BUILD ON)

# compile cuda files
file(GLOB CUDA_FILES cufiles/device.cu)
list(APPEND CUDA_NVCC_FLAGS --compiler-options -use_fast_math)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_53,code=sm_53) 
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60) 
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_62,code=sm_62) 
CUDA_COMPILE(cugen ${CUDA_FILES})


CUDA_COMPILE(cugen ${CUDA_FILES})


file(GLOB APP_SOURCES  src/*.cpp)
 

foreach( example ${APP_SOURCES} )
    set_source_files_properties( example  PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ )
    get_filename_component(srcfile ${example} NAME)
    message(STATUS ${srcfile})
    string( REPLACE ".cpp" "" filename ${srcfile} )        # remove .cpp extension
    set_source_files_properties( src/${filename} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ ) # set properties
    cuda_add_executable( ${filename} ${example} ${cugen})  # add executable
endforeach( example ${APP_SOURCES} )
 

# add flags for all C++ targets
list(APPEND CMAKE_CXX_FLAGS "-O3 -ffast-math -Wall -std=c++11")
 
