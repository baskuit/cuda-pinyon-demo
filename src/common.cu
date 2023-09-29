// #include <stdio.h>

#include <math.h>

#include "./common.hh"

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <curand.h>
#include <curand_kernel.h>

__device__ curandState *devStates;

__global__ void __convert_kernel(const uint64_t *input, float *output)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 47 * 8)
    {
        int input_index = tid / 8;
        int byte_index = tid % 8;
        uint64_t value = input[input_index];
        value >>= (64 - (byte_index * 8));
        value &= 0xFF;
        output[tid] = (float)value;
    }
}

__device__ float __generate(curandState* globalState, int ind)
{
    curandState localState = globalState[ind];
    float RANDOM = curand_uniform( &localState );
    globalState[ind] = localState;
    return RANDOM;
}

__global__ void __setup_kernel(unsigned long seed)
{
    cudaMalloc(&devStates, gridDim.x * sizeof(curandState));
    curand_init ( seed, blockIdx.x, 0, &(devStates[blockIdx.x]) );
}

__global__ void __sample_kernel(
    Buffers tgt,
    Buffers src,
    const int block_size,
    const int start_block_index,
    const int n_blocks)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int block_index = (start_block_index + blockIdx.x) % n_blocks;
    const int base_sample_index = block_index * block_size;
    const int sample_index = base_sample_index + tid;// (int)(ceil((__generate(devStates, blockIdx.x) * (block_size + 1))) - 1);
    memcpy(tgt.float_input_buffer + tid * 376, src.float_input_buffer + sample_index * 376, 376);
    // const int sample_index = base_sample_index + threadIdx.x;// + (int)(ceil((__generate(devStates, blockIdx.x) * (block_size + 1))) - 1);
    // memcpy(tgt.float_input_buffer + tid * 376, src.float_input_buffer + tid * 376, 376);
}

void sample(
    Buffers tgt,
    Buffers src,
    const int block_size,
    const int start_block_index,
    const int n_blocks,
    const int n_blocks_to_sample,
    const int n_samples_per_block)
{
    __sample_kernel<<<n_blocks_to_sample, n_samples_per_block>>>(tgt, src, block_size, start_block_index, n_blocks);
};

void convert(
    float *output,
    const uint64_t *input)
{
    __convert_kernel<<<12, 32>>>(input, output);
}

void copy(
    uint64_t *dest,
    const uint64_t *src,
    const int len)
{
    cudaMemcpy(dest, src, len, cudaMemcpyHostToDevice);
}

void alloc_pinned_buffers(
    Buffers &buffer_data,
    const long int batch_size)
{
    cudaMallocHost(&buffer_data.raw_input_buffer, batch_size * 47 * sizeof(uint64_t));
    cudaMallocHost(&buffer_data.float_input_buffer, batch_size * 376 * sizeof(float));
}

void alloc_device_buffers(
    Buffers &buffer_data,
    const long int batch_size)
{
    cudaMalloc(&buffer_data.raw_input_buffer, batch_size * 47 * sizeof(uint64_t));
    cudaMalloc(&buffer_data.float_input_buffer, batch_size * 376 * sizeof(float));
}

void dealloc_buffers(
    Buffers &buffer_data)
{
    cudaFree(buffer_data.raw_input_buffer);
    cudaFree(buffer_data.float_input_buffer);
}

void setup_rng(const int n)
{
    srand(time(0));
    int seed = rand();
    // blocks, 1 thread each
    __setup_kernel<<<n, 1>>>(seed);
}
