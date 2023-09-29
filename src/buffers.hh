#pragma once

#include <stddef.h>
#include <vector>

#include "./common.hh"

// RAII pinned buffer
struct PinnedBuffers : LearnerBuffers
{
    const size_t size = 0;

    PinnedBuffers() {}

    PinnedBuffers(const size_t size)
        : size{size}
    {
        alloc_pinned_buffers(*this, size);
    }

    ~PinnedBuffers()
    {
        dealloc_buffers(*this);
    }

    PinnedBuffers(const PinnedBuffers &) = delete;
    PinnedBuffers &operator=(const PinnedBuffers &) = delete;
};

// RAII pinned buffer
struct DeviceBuffers : LearnerBuffers
{
    const size_t size = 0;

    DeviceBuffers() {}

    DeviceBuffers(const size_t size)
        : size{size}
    {
        alloc_device_buffers(*this, size);
    }

    ~DeviceBuffers()
    {
        dealloc_buffers(*this);
    }

    DeviceBuffers(const DeviceBuffers &) = delete;
    DeviceBuffers &operator=(const DeviceBuffers &) = delete;
};

// Buffers local to actor threads for storing samples for training game in-progress
struct HostBuffers : ActorBuffers
{
    size_t size;
    std::vector<uint64_t> raw_input_vector{};
    std::vector<uint64_t> value_data_vector{};
    std::vector<float> joined_policy_vector{};
    std::vector<uint8_t> joined_actions_vector{};
    std::vector<uint32_t> joined_n_actions_vector{};

    HostBuffers(const size_t size)
        : size{size}
    {
        raw_input_vector.resize(size * 47);
        value_data_vector.resize(size * 1);
        joined_policy_vector.resize(size * 18);
        joined_actions_vector.resize(size * 18);
        joined_n_actions_vector.resize(size * 2);
        static_cast<ActorBuffers &>(*this) = ActorBuffers{
            raw_input_vector.data(),
            value_data_vector.data(),
            joined_policy_vector.data(),
            joined_actions_vector.data()};
    }

    HostBuffers(const HostBuffers &) = delete;
    // HostBuffers &operator=(const HostBuffers &) = delete;
};
