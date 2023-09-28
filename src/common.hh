#ifndef COMMON
#define COMMON

// __global__ void saxpy(int n, float a, float *x, float *y);

void convert(
    float *output,
    const unsigned long long *input,
    const int index);

void copy_battle(
    unsigned long long *ouput,
    const unsigned long long *battle_bytes,
    const int index);

void alloc_buffers(
    unsigned long long **raw_buffer,
    float **float_buffer,
    const long int batch_size);

void dealloc_buffers(
    unsigned long long **raw_buffer,
    float **float_buffer);

void cuda_test();

#endif
