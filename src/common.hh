#ifndef COMMON
#define COMMON

// __global__ void saxpy(int n, float a, float *x, float *y);

void convert(
    const unsigned long long *input,
    float *output);

void copy_battle(
    unsigned long long *host_ptr,
    const unsigned long long *battle_bytes,
    const int index);

void alloc_buffers(
    unsigned long long **host_ptr,
    float **host_ptr_float,
    const long int batch_size);

void dealloc_buffers(
    unsigned long long **host_ptr,
    float **host_ptr_float);

void cuda_test();

#endif
