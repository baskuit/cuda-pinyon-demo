#pragma once

#include <stdint.h>

const int n_bytes_battle = 376;
const int n_pokemon = 151;
const int n_moveslots = 165;
const int policy_size = n_pokemon + n_moveslots;

struct ActorBuffers
{
    uint64_t *raw_input_buffer;
    float *value_data_buffer;
    float *joined_policy_buffer;
    uint32_t *joined_policy_index_buffer;
};

struct LearnerBuffers
{
    float *float_input_buffer;
    float *value_data_buffer;
    float *joined_policy_buffer;
    uint32_t *joined_policy_index_buffer;
};

namespace CUDACommon
{
    void alloc_device_buffers(
        LearnerBuffers &buffer_data,
        const long int batch_size);

    void alloc_pinned_buffers(
        LearnerBuffers &buffer_data,
        const long int batch_size);

    void alloc_actor_buffers(
        ActorBuffers &buffer_data,
        const long int batch_size);

    void dealloc_buffers(
        LearnerBuffers &buffer_data);

    void dealloc_actor_buffers(
        ActorBuffers &buffer_data);

    void copy_game_to_sample_buffer(
        LearnerBuffers &sample_buffers,
        const ActorBuffers &actor_buffers,
        const int start_index,
        const int count,
        const int max_index);

    void copy_sample_to_learner_buffer(
        LearnerBuffers learner_buffer,
        const LearnerBuffers sample_buffer,
        const int start_index,
        const int count,
        const int n_samples,
        const int max_index);
};