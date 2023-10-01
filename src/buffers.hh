#pragma once

#include <stddef.h>
#include <vector>

#include "./common.hh"

// RAII pinned buffer
struct PinnedBuffers : LearnerBuffers
{
    PinnedBuffers() {}

    PinnedBuffers(const int size)
    {
        CUDACommon::alloc_pinned_buffers(*this, size);
    }

    ~PinnedBuffers()
    {
        CUDACommon::dealloc_buffers(*this);
    }

    PinnedBuffers(const PinnedBuffers &) = delete;
    PinnedBuffers &operator=(const PinnedBuffers &) = delete;
};

// RAII pinned buffer
struct DeviceBuffers : LearnerBuffers
{
    DeviceBuffers() {}

    DeviceBuffers(const int size)
    {
        CUDACommon::alloc_device_buffers(*this, size);
    }

    ~DeviceBuffers()
    {
        CUDACommon::dealloc_buffers(*this);
    }

    DeviceBuffers(const DeviceBuffers &) = delete;
    DeviceBuffers &operator=(const DeviceBuffers &) = delete;
};

// Buffers local to actor threads for storing samples for training game in-progress
struct HostBuffers : ActorBuffers
{
    int size;
    std::vector<uint64_t> raw_input_vector{};
    std::vector<float> value_data_vector{};
    std::vector<float> joined_policy_vector{};
    std::vector<uint8_t> joined_actions_vector{};

    HostBuffers(const int size)
        : size{size}
    {
        raw_input_vector.resize(size * 47);
        value_data_vector.resize(size * 2);
        joined_policy_vector.resize(size * 18);
        joined_actions_vector.resize(size * 18);
        static_cast<ActorBuffers &>(*this) = ActorBuffers{
            raw_input_vector.data(),
            value_data_vector.data(),
            joined_policy_vector.data(),
            joined_actions_vector.data()};
    }

    HostBuffers(const HostBuffers &) = delete;
    HostBuffers &operator=(const HostBuffers &) = delete;
};
