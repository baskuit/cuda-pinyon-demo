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
struct PinnedActorBuffers : ActorBuffers
{
    PinnedActorBuffers() {}

    PinnedActorBuffers(const int size)
    {
        CUDACommon::alloc_actor_buffers(*this, size);
    }

    ~PinnedActorBuffers()
    {
        CUDACommon::dealloc_actor_buffers(*this);
    }

    PinnedActorBuffers(const PinnedActorBuffers &) = delete;
    PinnedActorBuffers &operator=(const PinnedActorBuffers &) = delete;
};
