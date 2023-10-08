cmake_minimum_required(VERSION 3.8.0)
project(cuda-pinyon-demo LANGUAGES CXX CUDA)
set (CMAKE_CXX_STANDARD 20)

# find_package(CUDA REQUIRED)

set(CMAKE_PREFIX_PATH "/home/user/Desktop/libtorch/")
set(CMAKE_CUDA_ARCHITECTURES 61)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.1/bin/nvcc)
find_package(Torch REQUIRED)

# Add TORCH_USE_CUDA_DSA to CMAKE_CXX_FLAGS
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -DTORCH_USE_CUDA_DSA")

# nvcc flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61) # --allow-unsupported-compiler)

add_library(battle-stream STATIC
    src/cuda.cu
    src/cuda.hh
)

add_subdirectory(extern/pinyon)
target_include_directories(pinyon INTERFACE extern/pinyon/src)

include_directories(extern/engine/zig-out/include)

add_executable(main ./main.cc)
target_link_libraries(main battle-stream)
target_link_libraries(main ${PROJECT_SOURCE_DIR}/extern/engine/zig-out/lib/libpkmn-showdown.a)
target_link_libraries(main pinyon)
target_link_libraries(main "${TORCH_LIBRARIES}")