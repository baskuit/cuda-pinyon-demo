cmake_minimum_required(VERSION 3.1.0)
set (CMAKE_CXX_STANDARD 20)

project(cuda-pinyon-demo)

# packages                 
find_package(CUDA REQUIRED)   

option(ENABLE_TORCH_ "Enable Libtorch" ON)

if(ENABLE_TORCH_)
    set(CMAKE_PREFIX_PATH "/home/user/Desktop/libtorch/")
    set(CMAKE_CUDA_ARCHITECTURES 61)
    # change this depending on your CUDA version
    set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.1/bin/nvcc)
    # change this depending on installation path
    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
endif()


# nvcc flags                                                                                                                                                                                               
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61) # --allow-unsupported-compiler)

cuda_add_library(common src/common.cu)

add_subdirectory(extern/pinyon)
target_include_directories(pinyon INTERFACE extern/pinyon/src)

include_directories(extern/engine/zig-out/include)

add_executable(main main.cc)
target_link_libraries(main pinyon)
target_link_libraries(main ${PROJECT_SOURCE_DIR}/extern/engine/zig-out/lib/libpkmn-showdown.a)
target_link_libraries(main common)
if(ENABLE_TORCH_)
    target_link_libraries(main "${TORCH_LIBRARIES}")
    target_compile_definitions(main  PUBLIC ENABLE_TORCH)
endif()