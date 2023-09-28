cmake_minimum_required(VERSION 3.1.0)
set (CMAKE_CXX_STANDARD 20)

project(cuda-pinyon-demo)

# packages                 
find_package(CUDA)                                                                                                                                                                                
# find_package(CUDAToolkit)

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
