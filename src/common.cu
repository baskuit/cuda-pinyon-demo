// #include <stdio.h>

#include <math.h>

#include "./common.hh"

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <curand.h>
#include <curand_kernel.h>

namespace Kernels
{
    __global__ void convert_battle_bytes_to_floats(
        float *tgt,
        const uint64_t *src,
        const int start_index,
        const int count,
        const int max_index)
    {
        int game_index = blockIdx.x / 12;
        int byte_index = (blockIdx.x % 12) * 32 + threadIdx.x;
        if (byte_index < 376)
        {
            uint64_t byte_value = src[game_index * 47 + byte_index / 8];
            byte_value >>= (64 - (byte_index * 8));
            byte_value &= 0xFF;
            tgt[(start_index + game_index) * 376 + byte_index] = (float)byte_value;
        }
    }
    __global__ void __sample_kernel(
        LearnerBuffers tgt,
        LearnerBuffers src,
        const int block_size,
        const int start_block_index,
        const int n_blocks)
    {
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        const int block_index = (start_block_index + blockIdx.x) % n_blocks;
        const int base_sample_index = block_index * block_size;

        curandState state;
        curand_init(clock64(), tid, 0, &state);

        const int sample_index =
            base_sample_index +
            (int)(ceil((curand_uniform(&state) * (block_size + 1))) - 1);
        memset(
            tgt.float_input_buffer + tid * n_bytes_battle,
            1.4, n_bytes_battle);
        memcpy(
            tgt.value_data_buffer + tid * 8,
            src.value_data_buffer + base_sample_index * 8, 8);
        memcpy(
            tgt.joined_policy_buffer + tid * 18 * 4,
            src.joined_policy_buffer + base_sample_index * 18 * 4, 18 * 4);
        memcpy(
            tgt.joined_policy_index_buffer + tid * 18 * 4,
            src.joined_policy_index_buffer + base_sample_index * 18 * 4, 18 * 4);
    }
};

void CUDACommon::alloc_pinned_buffers(
    LearnerBuffers &buffer_data,
    const long int batch_size)
{
    cudaMallocHost(&buffer_data.float_input_buffer, batch_size * n_bytes_battle * sizeof(float));
    cudaMallocHost(&buffer_data.value_data_buffer, batch_size * 1 * sizeof(uint64_t));
    cudaMallocHost(&buffer_data.joined_policy_buffer, batch_size * 18 * sizeof(float));
    cudaMallocHost(&buffer_data.joined_policy_index_buffer, batch_size * 18 * sizeof(uint32_t));
}

void CUDACommon::alloc_device_buffers(
    LearnerBuffers &buffer_data,
    const long int batch_size)
{
    cudaMalloc(&buffer_data.float_input_buffer, batch_size * n_bytes_battle * sizeof(float));
    cudaMalloc(&buffer_data.value_data_buffer, batch_size * 2 * sizeof(float));
    cudaMalloc(&buffer_data.joined_policy_buffer, batch_size * 18 * sizeof(float));
    cudaMalloc(&buffer_data.joined_policy_index_buffer, batch_size * 18 * sizeof(uint32_t));
}

void CUDACommon::dealloc_buffers(
    LearnerBuffers &buffer_data)
{
    cudaFree(buffer_data.float_input_buffer);
    cudaFree(buffer_data.value_data_buffer);
    cudaFree(buffer_data.joined_policy_buffer);
    cudaFree(buffer_data.joined_policy_index_buffer);
}

void CUDACommon::copy_game_to_sample_buffer(
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
    // memcpy(sample_buffers.value_data_buffer, actor_buffers.value_data_buffer, 2 * c);
}

void CUDACommon::copy_sample_to_learner_buffer(
    LearnerBuffers learner_buffers,
    LearnerBuffers sample_buffers,
    const int start_index,
    const int range,
    const int n_samples)
{
}
