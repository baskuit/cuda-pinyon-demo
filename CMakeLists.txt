cmake_minimum_required(VERSION 3.1.0)
set (CMAKE_CXX_STANDARD 20)

project(cuda-pinyon-demo)

# packages                 
find_package(CUDA REQUIRED)   


set(CMAKE_PREFIX_PATH "/home/user/Desktop/libtorch/")
set(CMAKE_CUDA_ARCHITECTURES 61)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.1/bin/nvcc)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# nvcc flags                                                                                                                                                                                               
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61) # --allow-unsupported-compiler)

add_executable(main main.cu)
target_link_libraries(main "${TORCH_LIBRARIES}")