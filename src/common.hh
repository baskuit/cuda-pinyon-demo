#ifndef COMMON
#define COMMON

#include <stdint.h>

// __global__ void saxpy(int n, float a, float *x, float *y);

struct BufferData
{
    uint64_t *raw_input_buffer;
    float *float_input_buffer;
    uint64_t *value_data_tensor;
    float *joined_policy_buffer;
    unsigned int *joined_policy_index_buffer;
};

void convert(
    float *output,
    const uint64_t *input);

void copy(
    uint64_t *dest,
    const uint64_t *src,
    const int len);

void alloc_buffers(
    uint64_t **raw_buffer,
    float **float_buffer,
    const long int batch_size);

void dealloc_buffers(
    uint64_t **raw_buffer,
    float **float_buffer);

void cuda_test();

void sample(
    BufferData tgt,
    BufferData src,
    const int block_size,
    const int start_block_index,
    const int num_blocks_to_sample,
    const int num_samples_per_block);

void setup_rng(const int n_blocks);
#endif
