// #include <stdio.h>

#include "./common.hh"

__global__ void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

void alloc_buffers (
    float **host_ptr, 
    float** device_ptr, 
    const long int batch_size, 
    const long int input_size) {

}

void cuda_test()
{
    int N = 1 << 20;
    float *x, *y, *d_x, *d_y;
    x = (float *)malloc(N * sizeof(float));
    y = (float *)malloc(N * sizeof(float));

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
}
