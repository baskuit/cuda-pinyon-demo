#pragma once

#include <stdint.h>

const int n_bytes_input = 376;

struct ActorBuffers
{
    uint64_t *raw_input_buffer;
    uint64_t *value_data_buffer;
    float *joined_policy_buffer;
    uint8_t *joined_actions_buffer;
    uint32_t *joined_n_actions_buffer;
};

struct LearnerBuffers
{
    uint64_t *raw_input_buffer;
    float *float_input_buffer;
    uint64_t *value_data_buffer;
    float *joined_policy_buffer;
    uint32_t *joined_policy_index_buffer;
};

void convert(
    float *output,
    const uint64_t *input);

void copy(
    uint64_t *dest,
    const uint64_t *src,
    const int len);

void add_state_to_actor_buffers(
    ActorBuffers &buffers);

void alloc_pinned_buffers(
    LearnerBuffers &buffer_data,
    const long int batch_size);

void alloc_device_buffers(
    LearnerBuffers &buffer_data,
    const long int batch_size);

void dealloc_buffers(
    LearnerBuffers &buffer_data);

void sample(
    LearnerBuffers tgt,
    LearnerBuffers src,
    const int block_size,
    const int n_blocks,
    const int start_block_index,
    const int num_blocks_to_sample,
    const int num_samples_per_block);

void setup_rng(const int n_blocks);
