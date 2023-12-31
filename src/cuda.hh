#pragma once

#include <stdint.h>

const int n_bytes_battle = 376;
const int n_pokemon = 151;
const int n_moveslots = 165;
const int policy_size = 1 + n_pokemon + n_moveslots;
const size_t log_size = 64;

struct ActorBuffers
{
    uint64_t *raw_input_buffer;
    float *value_data_buffer;
    float *joined_policy_buffer;
    int64_t *joined_policy_index_buffer;
};

struct LearnerBuffers
{
    float *float_input_buffer;
    float *value_data_buffer;
    float *joined_policy_buffer;
    int64_t *joined_policy_index_buffer;
};

void switch_device(
    const char device_index
);

void alloc_actor_buffers(
    ActorBuffers &buffer_data,
    const long int batch_size);

void alloc_pinned_buffers(
    LearnerBuffers &buffer_data,
    const long int batch_size);

void alloc_device_buffers(
    LearnerBuffers &buffer_data,
    const long int batch_size);

void dealloc_buffers(
    LearnerBuffers &buffer_data);

void dealloc_actor_buffers(
    ActorBuffers &buffer_data);

void copy_sample_to_learn_buffer(
    LearnerBuffers learn_buffers,
    const LearnerBuffers sample_buffers,
    const int start_index,
    const int count,
    const int max_index,
    const int n_samples);

void copy_game_to_sample_buffer(
    LearnerBuffers &sample_buffers,
    const ActorBuffers &actor_buffers,
    const int start_index,
    const int count,
    const int max_index);
