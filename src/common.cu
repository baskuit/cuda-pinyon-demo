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

__global__ void __setup_kernel ( unsigned long seed )
{
    cudaMalloc(&devStates, gridDim.x * sizeof(curandState));
    curand_init ( seed, blockIdx.x, 0, &devStates[blockIdx.x] );
}

__global__ void __sample_kernel(
    BufferData tgt,
    BufferData src,
    const int block_size,
    const int start_block_index)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int block_index = start_block_index + blockIdx.x;
    const int base_sample_index = block_index * block_size;
    const int sample_index = base_sample_index + (int)(ceil((__generate(devStates, blockIdx.x) * (block_size + 1))) - 1);
    memcpy(tgt.float_input_buffer + tid * 376, src.float_input_buffer + sample_index * 376, 376);
}

void sample(
    BufferData tgt,
    BufferData src,
    const int block_size,
    const int start_block_index,
    const int num_blocks_to_sample,
    const int num_samples_per_block)
{
    __sample_kernel<<<num_blocks_to_sample, num_samples_per_block>>>(tgt, src, block_size, start_block_index);
};

void convert(
    float *output,
    const uint64_t *input)
{
    dim3 gridDim(12, 1, 1);
    dim3 blockDim(32, 1, 1);
    __convert_kernel<<<gridDim, blockDim>>>(input, output);
}

void copy(
    uint64_t *dest,
    const uint64_t *src,
    const int len)
{
    cudaMemcpy(dest, src, len, cudaMemcpyHostToDevice);
}

void alloc_buffers(
    uint64_t **raw_buffer,
    float **float_buffer,
    // float **joined_policy_buffer,
    const long int batch_size)
{

    cudaMallocHost(raw_buffer, batch_size * 47 * sizeof(uint64_t));
    cudaMallocHost(float_buffer, batch_size * 376 * sizeof(float));
    // cudaMallocHost(joined_policy_buffer, batch_size * 18 * sizeof(float));
}

void alloc_buffers2(
    uint64_t **raw_buffer,
    float **float_buffer,
    // float **joined_policy_buffer,
    const long int batch_size)
{

    cudaMallocHost(raw_buffer, batch_size * 47 * sizeof(uint64_t));
    cudaMallocHost(float_buffer, batch_size * 376 * sizeof(float));
    //
}

void dealloc_buffers(
    uint64_t **raw_buffer,
    float **float_buffer)
{
    cudaFree(raw_buffer);
    cudaFree(float_buffer);
}

void setup_rng(const int n_blocks)
{
    srand(time(0));
    int seed = rand();
    __setup_kernel<<<n_blocks, 1>>>(seed);
}
