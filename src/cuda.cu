#include "./cuda.hh"

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

void alloc_actor_buffers(
    ActorBuffers &buffer_data,
    const long int batch_size)
{
    cudaMallocHost(&buffer_data.raw_input_buffer, batch_size * 47 * sizeof(uint64_t));
    cudaMallocHost(&buffer_data.value_data_buffer, batch_size * 2 * sizeof(float));
    cudaMallocHost(&buffer_data.joined_policy_buffer, batch_size * 18 * sizeof(float));
    cudaMallocHost(&buffer_data.joined_policy_index_buffer, batch_size * 18 * sizeof(int64_t));
}

void alloc_pinned_buffers(
    LearnerBuffers &buffer_data,
    const long int batch_size)
{
    cudaMallocHost(&buffer_data.float_input_buffer, batch_size * n_bytes_battle * sizeof(float));
    cudaMallocHost(&buffer_data.value_data_buffer, batch_size * 2 * sizeof(float));
    cudaMallocHost(&buffer_data.joined_policy_buffer, batch_size * 18 * sizeof(float));
    cudaMallocHost(&buffer_data.joined_policy_index_buffer, batch_size * 18 * sizeof(int64_t));
}

void alloc_device_buffers(
    LearnerBuffers &buffer_data,
    const long int batch_size)
{
    cudaMalloc(&buffer_data.float_input_buffer, batch_size * n_bytes_battle * sizeof(float));
    cudaMalloc(&buffer_data.value_data_buffer, batch_size * 2 * sizeof(float));
    cudaMalloc(&buffer_data.joined_policy_buffer, batch_size * 18 * sizeof(float));
    cudaMalloc(&buffer_data.joined_policy_index_buffer, batch_size * 18 * sizeof(int64_t));
}

void dealloc_buffers(
    LearnerBuffers &buffer_data)
{
    cudaFree(buffer_data.float_input_buffer);
    cudaFree(buffer_data.value_data_buffer);
    cudaFree(buffer_data.joined_policy_buffer);
    cudaFree(buffer_data.joined_policy_index_buffer);
}

void dealloc_actor_buffers(
    ActorBuffers &buffer_data)
{
    cudaFree(buffer_data.raw_input_buffer);
    cudaFree(buffer_data.value_data_buffer);
    cudaFree(buffer_data.joined_policy_buffer);
    cudaFree(buffer_data.joined_policy_index_buffer);
}

void switch_device(
    const int device)
{
    cudaSetDevice(device);
}

namespace Kernels
{
    __global__ void convert_battle_bytes_to_floats(
        float *tgt,
        const uint64_t *src,
        const int start_index,
        const int count,
        const int max_index)
    {
        const int game_index = blockIdx.x / 12;
        const int tgt_start_index = (start_index + game_index) % max_index;
        const int byte_index = (blockIdx.x % 12) * 32 + threadIdx.x;
        if (byte_index < 376)
        {
            uint64_t byte_value = src[game_index * 47 + byte_index / 8];
            byte_value >>= (8 * (byte_index % 8));
            byte_value &= 0xFF;
            tgt[tgt_start_index * 376 + byte_index] = (float)(byte_value);
        }
    }

    __global__ void __sample_kernel(
        LearnerBuffers tgt,
        const LearnerBuffers src,
        const int start,
        const int count,
        const int max_index,
        const int n_samples)
    {
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < n_samples)
        {
            curandState state;
            curand_init(clock64(), tid, 0, &state);
            const int sample_index =
                (start +
                 (int)(ceil((curand_uniform(&state) * count)) - 1)) %
                max_index;
            memcpy(
                tgt.float_input_buffer + tid * n_bytes_battle,
                src.float_input_buffer + sample_index * n_bytes_battle, n_bytes_battle * sizeof(float));
            memcpy(
                tgt.value_data_buffer + tid * 2,
                src.value_data_buffer + sample_index * 2, 2 * sizeof(float));
            memcpy(
                tgt.joined_policy_buffer + tid * 18,
                src.joined_policy_buffer + sample_index * 18, 18 * sizeof(float));
            memcpy(
                &tgt.joined_policy_index_buffer[tid * 18],
                src.joined_policy_index_buffer + sample_index * 18, 18 * sizeof(int64_t));
        }
    }
};

void copy_sample_to_learn_buffer(
    LearnerBuffers learn_buffers,
    const LearnerBuffers sample_buffers,
    const int start_index,
    const int count,
    const int max_index,
    const int n_samples)
{
    const int n_blocks = ceil(n_samples / (float)32);
    Kernels::__sample_kernel<<<n_blocks, 32>>>(
        learn_buffers,
        sample_buffers,
        start_index, count, max_index, n_samples);
    cudaDeviceSynchronize();
}

void copy_game_to_sample_buffer_(
    LearnerBuffers &sample_buffers,
    const ActorBuffers &actor_buffers,
    const int start_index,
    const int offset,
    const int count,
    const int max_index)
{
    memcpy(
        &sample_buffers.value_data_buffer[2 * start_index],
        &actor_buffers.value_data_buffer[2 * offset],
        2 * count * sizeof(float));
    memcpy(
        &sample_buffers.joined_policy_buffer[18 * start_index],
        &actor_buffers.joined_policy_buffer[18 * offset],
        18 * count * sizeof(float));
    memcpy(
        &sample_buffers.joined_policy_index_buffer[18 * start_index],
        &actor_buffers.joined_policy_index_buffer[18 * offset],
        18 * count * sizeof(int64_t));
}

void copy_game_to_sample_buffer_2(
    LearnerBuffers &sample_buffers,
    const ActorBuffers &actor_buffers,
    const int start_index,
    const int count,
    const int max_index)
{
}

void copy_game_to_sample_buffer(
    LearnerBuffers &sample_buffers,
    const ActorBuffers &actor_buffers,
    const int start_index,
    const int count,
    const int max_index)
{
    Kernels::convert_battle_bytes_to_floats<<<count * 12, 32>>>(
        sample_buffers.float_input_buffer,
        actor_buffers.raw_input_buffer,
        start_index, count, max_index);
    const int end = start_index + count;
    if (end <= max_index)
    {
        memcpy(
            &sample_buffers.value_data_buffer[2 * start_index],
            actor_buffers.value_data_buffer,
            2 * count * sizeof(float));
        memcpy(
            &sample_buffers.joined_policy_buffer[18 * start_index],
            actor_buffers.joined_policy_buffer,
            18 * count * sizeof(float));
        memcpy(
            &sample_buffers.joined_policy_index_buffer[18 * start_index],
            actor_buffers.joined_policy_index_buffer,
            18 * count * sizeof(int64_t));
    }
    else
    {
        const int end_ = end % max_index;
        const int count_ = max_index - start_index;
        // initial_
        memcpy(
            sample_buffers.value_data_buffer,
            &actor_buffers.value_data_buffer[2 * count_],
            2 * end_ * sizeof(float));
        memcpy(
            sample_buffers.joined_policy_buffer,
            &actor_buffers.joined_policy_buffer[18 * count_],
            18 * end_ * sizeof(float));
        memcpy(
            sample_buffers.joined_policy_index_buffer,
            &actor_buffers.joined_policy_index_buffer[18 * count_],
            18 * end_ * sizeof(int64_t));
        // start - end
        memcpy(
            &sample_buffers.value_data_buffer[2 * start_index],
            actor_buffers.value_data_buffer,
            2 * count_ * sizeof(float));
        memcpy(
            &sample_buffers.joined_policy_buffer[18 * start_index],
            actor_buffers.joined_policy_buffer,
            18 * count_ * sizeof(float));
        memcpy(
            &sample_buffers.joined_policy_index_buffer[18 * start_index],
            actor_buffers.joined_policy_index_buffer,
            18 * count_ * sizeof(int64_t));
    }
    cudaDeviceSynchronize();
}
