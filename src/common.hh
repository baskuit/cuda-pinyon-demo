#ifndef COMMON
#define COMMON

// __global__ void saxpy(int n, float a, float *x, float *y);

void alloc_buffers (
    float **host_ptr, 
    float** device_ptr, 
    const long int batch_size, 
    const long int input_size);

void cuda_test();

#endif
