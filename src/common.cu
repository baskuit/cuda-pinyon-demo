// #include <stdio.h>

#include "./common.hh"

#include <cuda_runtime_api.h>
#include <cuda.h>

__global__ void convertBytesToFloats(const unsigned long long *input, float *output)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 47 * 8)
    {
        // Calculate the index into the input array (each uint64_t contains 8 bytes)
        int input_index = tid / 8;
        int byte_index = tid % 8;

        // Extract the byte from the input uint64_t
        unsigned long long value = input[input_index];
        value >>= (64 - (byte_index * 8));
        value &= 0xFF;

        output[tid] = (float)value;
    }
}

void convert(
    float *output,
    const unsigned long long *input,
    const int index)
{
    dim3 gridDim(12, 1, 1);
    dim3 blockDim(32, 1, 1);
    convertBytesToFloats<<<gridDim, blockDim>>>(input + (index * 376), output + (index * 47));
}

void copy_battle(
    unsigned long long *host_ptr,
    const unsigned long long *battle_bytes,
    const int index)
{
    cudaMemcpy(host_ptr + (index * 47), battle_bytes, 376, cudaMemcpyHostToDevice);
}

void alloc_buffers(
    unsigned long long **raw_buffer,
    float **float_buffer,
    // float **joined_policy_buffer,
    const long int batch_size)
{

    cudaMallocHost(raw_buffer, batch_size * 47 * sizeof(unsigned long long));
    cudaMallocHost(float_buffer, batch_size * 376 * sizeof(float));
    // cudaMallocHost(joined_policy_buffer, batch_size * 18 * sizeof(float));
}

void dealloc_buffers(
    unsigned long long **raw_buffer,
    float **float_buffer)
{
    cudaFree(raw_buffer);
    cudaFree(float_buffer);
}
