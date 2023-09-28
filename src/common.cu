// #include <stdio.h>

#include "./common.hh"

#include <cuda_runtime_api.h>
#include <cuda.h>

__global__ void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

__global__ void convertBytesToFloats(const unsigned long long *input, float *output)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 48 * 8)
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

void convert(const unsigned long long *input, float *output)
{
    dim3 gridDim(12, 1, 1);
    dim3 blockDim(32, 1, 1);
    convertBytesToFloats<<<gridDim, blockDim>>>(input, output);
}

void copy_battle(
    unsigned long long *host_ptr,
    const unsigned long long *battle_bytes,
    const int index)
{
    cudaMemcpy(host_ptr + (index * 48), battle_bytes, 384, cudaMemcpyHostToDevice);
}

void alloc_buffers(
    unsigned long long **host_ptr,
    float **host_ptr_float,
    const long int batch_size)
{

    cudaMallocHost(host_ptr, batch_size * 48 * sizeof(unsigned long long));
    cudaMallocHost(host_ptr_float, batch_size * 384 * sizeof(float));
}

void dealloc_buffers(
    unsigned long long **host_ptr,
    float **host_ptr_float)
{
    cudaFree(host_ptr);
    cudaFree(host_ptr_float);
}
