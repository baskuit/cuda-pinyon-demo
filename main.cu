#include <torch/torch.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include <curand.h>
#include <curand_kernel.h>

#include <thread>
#include <iostream>

#include <math.h>
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
    int64_t *joined_policy_index_buffer;
};

struct LearnerBuffers
{
    float *float_input_buffer;
    float *value_data_buffer;
    float *joined_policy_buffer;
    int64_t *joined_policy_index_buffer;
};

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
        LearnerBuffers index_buffers,
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
            index_buffers.joined_policy_index_buffer[tid] = sample_index;
        }
    }
};

void copy_sample_to_learner_buffer(
    LearnerBuffers learner_buffers,
    const LearnerBuffers sample_buffers,
    LearnerBuffers index_buffers,
    const int start_index,
    const int count,
    const int max_index,
    const int n_samples)
{
    const int n_blocks = ceil(n_samples / (float)32);

    Kernels::__sample_kernel<<<n_blocks, 32>>>(
        learner_buffers,
        sample_buffers,
        index_buffers,
        start_index, count, max_index, n_samples);
    cudaDeviceSynchronize();
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
    const int initial = start_index + count;
    if (initial <= max_index)
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
        const int initial_ = initial % max_index;
        const int count_ = max_index - start_index;
        // initial_
        memcpy(
            sample_buffers.value_data_buffer,
            &actor_buffers.value_data_buffer[2 * count_],
            2 * initial_ * sizeof(float));
        memcpy(
            sample_buffers.joined_policy_buffer,
            &actor_buffers.joined_policy_buffer[18 * count_],
            18 * initial_ * sizeof(float));
        memcpy(
            sample_buffers.joined_policy_index_buffer,
            &actor_buffers.joined_policy_index_buffer[18 * count_],
            18 * initial_ * sizeof(int64_t));
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

struct PinnedBuffers : LearnerBuffers
{
    PinnedBuffers() {}

    PinnedBuffers(const int size)
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

struct DeviceBuffers : LearnerBuffers
{
    DeviceBuffers() {}

    DeviceBuffers(const int size)
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

struct PinnedActorBuffers : ActorBuffers
{
    PinnedActorBuffers() {}

    PinnedActorBuffers(const int size)
    {
        alloc_actor_buffers(*this, size);
    }

    ~PinnedActorBuffers()
    {
        dealloc_actor_buffers(*this);
    }

    PinnedActorBuffers(const PinnedActorBuffers &) = delete;
    PinnedActorBuffers &operator=(const PinnedActorBuffers &) = delete;
};

// print tensor
void pt(torch::Tensor tensor)
{
    int a = tensor.size(0);
    int b = tensor.size(1);
    std::cout << "size: " << a << ' ' << b << std::endl;
    for (int i = 0; i < a; ++i)
    {
        for (int j = 0; j < b; ++j)
        {
            std::cout << tensor.index({i, j}).item().toFloat() << " ";
        }
        std::cout << std::endl;
    }
}

namespace Options
{
    const int hidden_size = 1 << 7;
    const int input_size = 376;
    const int outer_size = 1 << 5;
    const int policy_size = 151 + 165;
    const int n_res_blocks = 4;
};

class ResBlock : public torch::nn::Module
{
public:
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};

    ResBlock()
    {
        fc1 = register_module("fc1", torch::nn::Linear(Options::hidden_size, Options::hidden_size));
        fc2 = register_module("fc2", torch::nn::Linear(Options::hidden_size, Options::hidden_size));
    }

    torch::Tensor forward(torch::Tensor input)
    {
        torch::Tensor residual = input.clone();
        input = torch::relu(fc2(torch::relu(fc1(input)))) + residual;
        return input;
    }
};

struct NetOutput
{
    torch::Tensor value, row_policy, col_policy;
};

class Net : public torch::nn::Module
{
public:
    torch::Device device = torch::kCUDA;
    torch::nn::Linear fc{nullptr};
    torch::nn::Sequential tower{};
    torch::nn::Linear fc_value_pre{nullptr};
    torch::nn::Linear fc_value{nullptr};
    torch::nn::Linear fc_row_logits_pre{nullptr};
    torch::nn::Linear fc_row_logits{nullptr};
    torch::nn::Linear fc_col_logits_pre{nullptr};
    torch::nn::Linear fc_col_logits{nullptr};

    Net()
    {
        // input
        fc = register_module("fc_input", torch::nn::Linear(Options::input_size, Options::hidden_size));
        // tower
        for (int i = 0; i < Options::n_res_blocks; ++i)
        {
            auto block = std::make_shared<ResBlock>();
            tower->push_back(register_module("b" + std::to_string(i), block));
        }
        // register_module("tower", tower);
        // value
        fc_value_pre = register_module("fc_value_pre", torch::nn::Linear(Options::hidden_size, Options::outer_size));
        fc_value = register_module("fc_value", torch::nn::Linear(Options::outer_size, 1));
        // policy
        fc_row_logits_pre = register_module("fc_row_logits_pre", torch::nn::Linear(Options::hidden_size, Options::outer_size));
        fc_row_logits = register_module("fc_row_logits", torch::nn::Linear(Options::outer_size, policy_size));
        fc_col_logits_pre = register_module("fc_col_logits_pre", torch::nn::Linear(Options::hidden_size, Options::outer_size));
        fc_col_logits = register_module("fc_col_logits", torch::nn::Linear(Options::outer_size, policy_size));
    }

    void to(const torch::Device &device)
    {
        static_cast<torch::nn::Module *>(this)->to(device);
        this->device = device;
    }

    NetOutput forward(torch::Tensor input, torch::Tensor joined_policy_indices)
    {
        torch::Tensor tower_ = tower->forward(torch::relu(fc(input)));
        torch::Tensor row_logits = fc_row_logits(torch::relu(fc_row_logits_pre(tower_)));
        torch::Tensor col_logits = fc_col_logits(torch::relu(fc_col_logits_pre(tower_)));
        torch::Tensor row_policy_indices = joined_policy_indices.index({"...", torch::indexing::Slice{0, 9, 1}});
        torch::Tensor col_policy_indices = joined_policy_indices.index({"...", torch::indexing::Slice{9, 18, 1}});
        torch::Tensor row_logits_picked = torch::gather(row_logits, 1, row_policy_indices);
        torch::Tensor col_logits_picked = torch::gather(col_logits, 1, col_policy_indices);
        torch::Tensor r = torch::log_softmax(row_logits_picked, 1);
        torch::Tensor c = torch::log_softmax(col_logits_picked, 1);
        torch::Tensor value = torch::sigmoid(fc_value(torch::relu(fc_value_pre(tower_))));
        return {value, r, c};
    }
};

template <typename Net>
struct Training
{
    Net net;

    bool train = true;
    bool generate_samples = true;

    const int sample_buffer_size;
    const int learner_buffer_size;

    PinnedBuffers sample_buffers{
        sample_buffer_size};
    std::atomic<uint64_t> sample_index{0};

    DeviceBuffers learner_buffers{
        learner_buffer_size};

    PinnedBuffers index_buffers{
        learner_buffer_size};

    std::mutex sample_mutex{};
    const int berth = 1 << 6;

    template <typename... Args>
    Training(
        Args... args,
        const int sample_buffer_size,
        const int learner_buffer_size)
        : net{args...},
          sample_buffer_size{sample_buffer_size},
          learner_buffer_size{learner_buffer_size}
    {
        net.to(torch::kCUDA);
        for (int i = 0; i < sample_buffer_size; ++i)
        {
            for (int j = 0; j < 18; ++j)
            {
                sample_buffers.joined_policy_index_buffer[i * 18 + j] = int64_t{i % 100};
            }
        }
    }

    Training(const Training &) = delete;

    void actor_store(const PinnedActorBuffers &actor_buffers, const int count)
    {
        const uint64_t s = sample_index.fetch_add(count);
        const int sample_index_first = s % sample_buffer_size;

        sample_mutex.lock();
        copy_game_to_sample_buffer(
            sample_buffers,
            actor_buffers,
            sample_index_first,
            count,
            sample_buffer_size);
        sample_mutex.unlock();
    }

    void learner_fetch()
    {
        const int start = (sample_index.load() + berth) % sample_buffer_size;
        int range = sample_buffer_size - 2 * berth;

        sample_mutex.lock();
        copy_sample_to_learner_buffer(
            learner_buffers,
            sample_buffers,
            index_buffers,
            start,
            range,
            sample_buffer_size,
            learner_buffer_size);
        sample_mutex.unlock();
    };

    void actor()
    {
        PinnedActorBuffers buffers{500};
        int buffer_index = 0;

        uint8_t battle_bytes[n_bytes_battle];
        for (int i = 0; i < n_bytes_battle; ++i)
        {
            battle_bytes[i] = uint8_t{i % 151};
        }

        std::vector<float> row_strategy{};
        row_strategy.resize(9);
        for (int action = 0; action < 9; ++action)
        {
            row_strategy[action] = 1 / 9.0f;
        }
        int rows = 9;
        int cols = 9;

        while (generate_samples)
        {
            for (int i = 0; i < 50; ++i)
            {
                // battle
                memcpy(
                    &buffers.raw_input_buffer[buffer_index * 47],
                    reinterpret_cast<const uint64_t *>(battle_bytes),
                    376);
                // value
                buffers.value_data_buffer[buffer_index * 2] = .314;
                // policy
                memcpy(
                    &buffers.joined_policy_buffer[buffer_index * 18],
                    row_strategy.data(),
                    rows * 4);
                memcpy(
                    &buffers.joined_policy_buffer[buffer_index * 18 + 9],
                    row_strategy.data(),
                    cols * 4);

                for (int i = 0; i < 18; ++i)
                {
                    buffers.joined_policy_index_buffer[buffer_index * 18 + i] = int64_t{i};
                }
                ++buffer_index;
            }
            sleep(1);
            actor_store(buffers, buffer_index);
            buffer_index = 0;
        }
    }

    void learner()
    {
        torch::optim::SGD optimizer{net.parameters(), .001};
        torch::nn::MSELoss mse{};
        torch::nn::CrossEntropyLoss cel{};

        while (train)
        {
            // copy over entire learner buffer with selections from sample buffer
            learner_fetch();

            torch::Tensor float_input =
                torch::from_blob(
                    learner_buffers.float_input_buffer,
                    {learner_buffer_size, n_bytes_battle})
                    .to(net.device);
            torch::Tensor value_data =
                torch::from_blob(
                    learner_buffers.value_data_buffer,
                    {learner_buffer_size, 2})
                    .to(net.device);
            torch::Tensor value_target = value_data.index({"...", torch::indexing::Slice{0, 1, 1}});
            torch::Tensor joined_policy_indices =
                torch::from_blob(
                    learner_buffers.joined_policy_index_buffer,
                    {learner_buffer_size, 18}, {torch::kInt64})
                    .to(torch::kCUDA);
            torch::Tensor joined_policy_target =
                torch::from_blob(
                    learner_buffers.joined_policy_buffer,
                    {learner_buffer_size, 18})
                    .to(net.device);
            torch::cuda::synchronize();

            // dummy inputs; should work
            // torch::Tensor float_input = torch::rand({learner_buffer_size, 376}).to(torch::kCUDA);
            // torch::Tensor value_target = torch::rand({learner_buffer_size, 1}).to(torch::kCUDA);
            // torch::Tensor joined_policy_indices = torch::ones({learner_buffer_size, 18}, torch::kInt64).to(torch::kCUDA);
            // torch::Tensor joined_policy_target = torch::rand({learner_buffer_size, 18}).to(torch::kCUDA);
            // std::cout << "entire learner policy index " << std::endl;
            // pt(jpi);

            torch::Tensor row_policy_target = joined_policy_target.index({"...", torch::indexing::Slice{0, 9, 1}});
            torch::Tensor col_policy_target = joined_policy_target.index({"...", torch::indexing::Slice{9, 18, 1}});
            row_policy_target /= row_policy_target.norm(1, {1}, true);
            col_policy_target /= col_policy_target.norm(1, {1}, true);

            auto output = net.forward(float_input, joined_policy_indices);

            torch::Tensor value_loss =
                mse(output.value, value_target);
            // torch::Tensor row_policy_loss =
            //     torch::nn::functional::kl_div(
            //         output.row_policy,
            //         row_policy_target,
            //         torch::nn::functional::KLDivFuncOptions(torch::kBatchMean));
            // torch::Tensor col_policy_loss =
            //     torch::nn::functional::kl_div(
            //         output.col_policy,
            //         col_policy_target,
            //         torch::nn::functional::KLDivFuncOptions(torch::kBatchMean));
            torch::Tensor loss = value_loss;// + row_policy_loss + col_policy_loss;
            loss.backward();
            optimizer.step();
            optimizer.zero_grad();
            pt(output.value);
            pt(value_target);
            sleep(2);
            std::cout << "loss: " << loss.item().toFloat() << std::endl;
        }
    }

    void start(const int n_actor_threads)
    {
        std::thread actor_threads[n_actor_threads];
        for (int i = 0; i < n_actor_threads; ++i)
        {
            actor_threads[i] = std::thread(&Training<Net>::actor, this);
        }
        std::thread learner_thread{&Training<Net>::learner, this};
        for (int i = 0; i < n_actor_threads; ++i)
        {
            actor_threads[i].join();
        }
        learner_thread.join();
    }
};

int main()
{
    const int sample_buffer_size = 1 << 16;
    const int learner_minibatch_size = 1 << 4;
    Training<Net> training_workspace{sample_buffer_size, learner_minibatch_size};
    training_workspace.start(2);

    return 0;
}
