#include <cuda_runtime_api.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include <torch/torch.h>

#include <pinyon.hh>
#include <pkmn.h>

#include <queue>

#include "./sides.hh"

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

namespace Options
{
    const int hidden_size = 1 << 7;
    const int input_size = 376;
    const int outer_size = 1 << 5;
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

class NetImpl : public torch::nn::Module
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

    NetImpl()
    {
        fc = register_module("fc_input", torch::nn::Linear(Options::input_size, Options::hidden_size));
        for (int i = 0; i < Options::n_res_blocks; ++i)
        {
            auto block = std::make_shared<ResBlock>();
            tower->push_back(register_module("b" + std::to_string(i), block));
        }
        fc_value_pre = register_module("fc_value_pre", torch::nn::Linear(Options::hidden_size, Options::outer_size));
        fc_value = register_module("fc_value", torch::nn::Linear(Options::outer_size, 1));
        fc_row_logits_pre = register_module("fc_row_logits_pre", torch::nn::Linear(Options::hidden_size, Options::outer_size));
        fc_row_logits = register_module("fc_row_logits", torch::nn::Linear(Options::outer_size, policy_size - 1));
        fc_col_logits_pre = register_module("fc_col_logits_pre", torch::nn::Linear(Options::hidden_size, Options::outer_size));
        fc_col_logits = register_module("fc_col_logits", torch::nn::Linear(Options::outer_size, policy_size - 1));
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
        // torch::Tensor row_logits_picked = torch::gather(row_logits, 1, row_policy_indices);
        // torch::Tensor col_logits_picked = torch::gather(col_logits, 1, col_policy_indices);
        torch::Tensor neg_inf = -1 * (1 << 10) * torch::ones({input.size(0), 1}, torch::kInt64).to(torch::kCUDA);
        torch::Tensor row_logits_picked = torch::gather(torch::cat({neg_inf, row_logits}, 1), 1, row_policy_indices);
        torch::Tensor col_logits_picked = torch::gather(torch::cat({neg_inf, col_logits}, 1), 1, col_policy_indices);
        torch::Tensor r = torch::log_softmax(row_logits_picked, 1);
        torch::Tensor c = torch::log_softmax(col_logits_picked, 1);
        torch::Tensor value = torch::sigmoid(fc_value(torch::relu(fc_value_pre(tower_))));
        return {value, r, c};
    }
};
TORCH_MODULE(Net);

using TypeList = DefaultTypes<
    float,
    pkmn_choice,
    std::array<uint8_t, log_size>,
    bool,
    ConstantSum<1, 1>::Value,
    A<9>::Array>;

struct BattleTypes : TypeList
{

    class State : public PerfectInfoState<TypeList>
    {
    public:
        pkmn_gen1_battle battle;
        pkmn_gen1_log_options log_options;
        pkmn_gen1_battle_options options{};
        pkmn_result result;

        State(const uint64_t seed = 0)
        {
            TypeList::PRNG device{seed};
            const auto row_side = sides[device.random_int(n_sides)];
            const auto col_side = sides[device.random_int(n_sides)];
            memcpy(battle.bytes, row_side, 184);
            memcpy(battle.bytes + 184, col_side, 184);
            for (int i = 2 * 184; i < n_bytes_battle; ++i)
            {
                battle.bytes[i] = 0;
            }
            log_options = {this->obs.get().data(), log_size};
            pkmn_gen1_battle_options_set(&options, &log_options, NULL, NULL);
            get_actions();
        }

        State(const State &other)
        {
            this->row_actions = other.row_actions;
            this->col_actions = other.col_actions;
            this->terminal = other.terminal;
            // memcpy(battle.bytes, other.battle.bytes, 384 - 8); // don't need seed
            memcpy(battle.bytes, other.battle.bytes, 384);
            log_options = {this->obs.get().data(), log_size};
            pkmn_gen1_battle_options_set(&options, &log_options, NULL, NULL);
        }

        State &operator=(const State &other)
        {
            this->row_actions = other.row_actions;
            this->col_actions = other.col_actions;
            this->terminal = other.terminal;
            memcpy(battle.bytes, other.battle.bytes, 384);
            log_options = {this->obs.get().data(), log_size};
            pkmn_gen1_battle_options_set(&options, &log_options, NULL, NULL);
            return *this;
        }

        void get_actions()
        {
            this->row_actions.resize(
                pkmn_gen1_battle_choices(
                    &battle,
                    PKMN_PLAYER_P1,
                    pkmn_result_p1(result),
                    reinterpret_cast<pkmn_choice *>(this->row_actions.data()),
                    PKMN_MAX_CHOICES));
            this->col_actions.resize(
                pkmn_gen1_battle_choices(
                    &battle,
                    PKMN_PLAYER_P2,
                    pkmn_result_p2(result),
                    reinterpret_cast<pkmn_choice *>(this->col_actions.data()),
                    PKMN_MAX_CHOICES));
        }

        void apply_actions(
            TypeList::Action row_action,
            TypeList::Action col_action)
        {
            result = pkmn_gen1_battle_update(&battle, row_action.get(), col_action.get(), &options);
            const pkmn_result_kind result_kind = pkmn_result_type(result);
            if (result_kind) [[unlikely]]
            {
                this->terminal = true;
                if (result_kind == PKMN_RESULT_WIN)
                {
                    this->payoff = TypeList::Value{1.0f};
                }
                else if (result_kind == PKMN_RESULT_LOSE)
                {
                    this->payoff = TypeList::Value{0.0f};
                }
                else
                {
                    this->payoff = TypeList::Value{0.5f};
                }
            }
            else [[likely]]
            {
                pkmn_gen1_battle_options_set(&options, NULL, NULL, NULL);
            }
        }

        void randomize_transition(TypeList::PRNG &device)
        {
            uint8_t *battle_prng_bytes = battle.bytes + n_bytes_battle;
            *(reinterpret_cast<uint64_t *>(battle_prng_bytes)) = device.get_seed();
        }
    };
};

// Pinyon type list for the entire program: Single threaded search on battles using Monte-Carlo eval.
using Types = TreeBandit<Exp3<MonteCarloModel<BattleTypes>>, FlatNodes>;
// using Types = FlatSearch<Exp3<MonteCarloModel<BattleTypes>>>;

struct Metric
{
    std::mutex mtx{};

    const int max_donations;
    Metric(const int max_donations) : max_donations{max_donations}
    {
    }

    std::queue<int> donation_sizes;
    std::queue<decltype(std::chrono::high_resolution_clock::now())> donation_times;
    int total_donations = 0;

    float update_and_get_rate(const int donation_size)
    {
        mtx.lock();
        int count = donation_sizes.size() + 1;
        total_donations += donation_size;
        auto time = std::chrono::high_resolution_clock::now();

        donation_sizes.push(donation_size);
        donation_times.push(time);
        if (count > max_donations)
        {
            total_donations -= donation_sizes.front();
            donation_sizes.pop();
            donation_times.pop();
            --count;
        }

        const float duration = std::chrono::duration_cast<std::chrono::milliseconds>(time - donation_times.front()).count();
        if (duration == 0)
        {
            mtx.unlock();
            return 0;
        }
        const float answer = 1000 * (total_donations - donation_sizes.front()) / duration;
        mtx.unlock();
        return answer;
    }
};

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

    const size_t full_iterations = 1 << 10;
    const size_t partial_iterations = 1 << 8;
    const float full_search_prob = .25;
    const int berth = 1 << 8;

    std::mutex sample_mutex{};

    Metric actor_metric{100};
    Metric learn_metric{100};

    template <typename... Args>
    Training(
        Args... args,
        const int sample_buffer_size,
        const int learner_buffer_size)
        : net{args...},
          sample_buffer_size{sample_buffer_size},
          learner_buffer_size{learner_buffer_size}
    {
        net->to(torch::kCUDA);
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

        float rate = actor_metric.update_and_get_rate(count);
        if (rate > 0)
        {
            std::cout << "actor rate: " << rate << " count: " << count << std::endl;
        }

        // sample_mutex.lock();
        copy_game_to_sample_buffer(
            sample_buffers,
            actor_buffers,
            sample_index_first,
            count,
            sample_buffer_size);
        // sample_mutex.unlock();

        if (s > sample_buffer_size)
        {
            if (!train)
            {
                train = true;
                std::cout << "TRAINING ENABLED" << std::endl;
            }
        }
    }

    void learner_fetch()
    {
        const int start = (sample_index.load() + berth) % sample_buffer_size;
        // sample_mutex.lock();
        copy_sample_to_learner_buffer(
            learner_buffers,
            sample_buffers,
            index_buffers,
            start,
            sample_buffer_size - 2 * berth,
            sample_buffer_size,
            learner_buffer_size);
        // sample_mutex.unlock();
    };

    void actor()
    {
        PinnedActorBuffers buffers{500};
        int buffer_index = 0;
        Types::PRNG device{};
        Types::State state{device.get_seed()};
        Types::Model model{0};
        Types::Search search{};
        int rows, cols;
        Types::Value value;
        Types::VectorReal row_strategy, col_strategy;
        while (generate_samples)
        {
            if (state.is_terminal())
            {
                for (int i = 0; i < buffer_index; ++i)
                {
                    buffers.value_data_buffer[2 * i + 1] = state.payoff.get_row_value().get();
                }
                actor_store(buffers, buffer_index);
                buffer_index = 0;
                state = Types::State{device.get_seed()};
                state.randomize_transition(device);
            }
            rows = state.row_actions.size();
            cols = state.col_actions.size();
            row_strategy.clear();
            col_strategy.clear();
            if (rows == 1 && cols == 1)
            {
                state.apply_actions(
                    state.row_actions[0],
                    state.col_actions[0]);
                state.get_actions();
                continue;
            }
            Types::MatrixNode root{};
            const bool use_full_search = device.uniform() < full_search_prob;
            const size_t iterations = use_full_search ? full_iterations : partial_iterations;
            // search.run_for_iterations(iterations, device, state, model);
            search.run_for_iterations(iterations, device, state, model, root);
            // const Types::MatrixStats &stats = search.matrix_stats[0];
            const Types::MatrixStats &stats = root.stats;
            search.get_empirical_strategies(stats, row_strategy, col_strategy);
            search.get_empirical_value(stats, value);
            const int row_idx = device.random_int(rows);
            const int col_idx = device.random_int(cols);
            if (use_full_search)
            {
                memcpy(
                    &buffers.raw_input_buffer[buffer_index * 47],
                    reinterpret_cast<const uint64_t *>(state.battle.bytes),
                    376);
                buffers.value_data_buffer[buffer_index * 2] = value.get_row_value().get();
                memcpy(
                    &buffers.joined_policy_buffer[buffer_index * 18],
                    reinterpret_cast<float *>(row_strategy.data()),
                    rows * 4);
                for (int i = rows; i < 9; ++i)
                {
                    buffers.joined_policy_buffer[buffer_index * 18 + i] = 0.0f;
                }
                memcpy(
                    &buffers.joined_policy_buffer[buffer_index * 18 + 9],
                    reinterpret_cast<float *>(col_strategy.data()),
                    cols * 4);
                for (int i = cols; i < 9; ++i)
                {
                    buffers.joined_policy_buffer[buffer_index * 18 + 9 + i] = 0.0f;
                }
                auto get_index_from_action = [](const uint8_t *bytes, const uint8_t choice, const uint8_t col_offset = 0)
                {
                    uint8_t type = choice & 3;
                    uint8_t data = choice >> 2;
                    if (type == 1)
                    {
                        uint8_t moveid = bytes[2 * (data - 1) + 10 + col_offset]; // 0 - 165
                        return int64_t{moveid};                                   // index=0 is dummy very negative logit
                    }
                    else if (type == 2)
                    {
                        uint8_t slot = bytes[176 + data - 1 + col_offset];
                        int dex = bytes[24 * (slot - 1) + 21 + col_offset]; // 0 - 151
                        return int64_t{dex + 165};
                    }
                    else
                    {
                        return int64_t{0};
                    }
                };
                for (int i = 0; i < 9; ++i)
                {
                    if (i < rows)
                    {
                        buffers.joined_policy_index_buffer[buffer_index * 18 + i] = get_index_from_action(state.battle.bytes, state.row_actions[i].get(), 0);
                    }
                    else
                    {
                        buffers.joined_policy_index_buffer[buffer_index * 18 + i] = 0;
                    }
                }
                if (buffers.joined_policy_index_buffer[buffer_index * 18] == 0 && rows == 1)
                {
                    buffers.joined_policy_index_buffer[buffer_index * 18] = 1;
                }
                for (int i = 0; i < 9; ++i)
                {
                    if (i < cols)
                    {
                        buffers.joined_policy_index_buffer[buffer_index * 18 + 9 + i] = get_index_from_action(state.battle.bytes, state.col_actions[i].get(), 184);
                    }
                    else
                    {
                        buffers.joined_policy_index_buffer[buffer_index * 18 + 9 + i] = 0;
                    }
                }
                if (buffers.joined_policy_index_buffer[buffer_index * 18 + 9] == 0 && cols == 1)
                {
                    buffers.joined_policy_index_buffer[buffer_index * 18 + 9] = 1;
                }
                ++buffer_index;
            }
            state.apply_actions(state.row_actions[row_idx], state.col_actions[col_idx]);
            state.get_actions();
        }
    }

    void learner()
    {
        torch::optim::SGD optimizer{net->parameters(), .001};
        torch::nn::MSELoss mse{};
        torch::nn::CrossEntropyLoss cel{};

        while (!train)
        {
            sleep(1);
        }

        size_t checkpoint = 0;
        while (train)
        {
            for (size_t step = 0; step < (1 << 7); ++step)
            {

                learner_fetch();

                torch::Tensor float_input =
                    torch::from_blob(
                        learner_buffers.float_input_buffer,
                        {learner_buffer_size, n_bytes_battle})
                        .to(net->device);
                torch::Tensor value_data =
                    torch::from_blob(
                        learner_buffers.value_data_buffer,
                        {learner_buffer_size, 2})
                        .to(net->device);
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
                        .to(net->device);
                torch::cuda::synchronize();

                torch::Tensor row_policy_target = joined_policy_target.index({"...", torch::indexing::Slice{0, 9, 1}});
                torch::Tensor col_policy_target = joined_policy_target.index({"...", torch::indexing::Slice{9, 18, 1}});
                row_policy_target /= row_policy_target.norm(1, {1}, true);
                col_policy_target /= col_policy_target.norm(1, {1}, true);

                NetOutput output = net->forward(float_input, joined_policy_indices);

                torch::Tensor value_loss =
                    mse(output.value, value_target);
                torch::Tensor row_policy_loss =
                    torch::nn::functional::kl_div(
                        output.row_policy,
                        row_policy_target,
                        torch::nn::functional::KLDivFuncOptions(torch::kBatchMean));
                torch::Tensor col_policy_loss =
                    torch::nn::functional::kl_div(
                        output.col_policy,
                        col_policy_target,
                        torch::nn::functional::KLDivFuncOptions(torch::kBatchMean));
                torch::Tensor loss = value_loss + row_policy_loss + col_policy_loss;
                loss.backward();
                optimizer.step();
                optimizer.zero_grad();
                pt(row_policy_target.index({torch::indexing::Slice(0, 4, 1), "..."}));
                pt(output.row_policy.index({torch::indexing::Slice(0, 4, 1), "..."}));
                // std::this_thread::sleep_for(std::chrono::milliseconds(200));
                // std::cout << "loss: " << loss.item().toFloat() << std::endl;
                float rate = learn_metric.update_and_get_rate(1);

                while (rate > 20)
                {
                    sleep(1);
                    rate = learn_metric.update_and_get_rate(0);
                }
            }

            {
                auto timestamp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                std::ostringstream oss;
                oss << std::put_time(std::gmtime(&timestamp), "%Y%m%d%H%M%S");
                std::string timestampStr = oss.str();
                std::string filename = "../saved/model_" + timestampStr + ".pt";
                torch::save(net, filename);
            }

            ++checkpoint;
        }
    }

    void start(const int n_actor_threads)
    {
        std::thread actor_threads[n_actor_threads];
        for (int i = 0; i < n_actor_threads; ++i)
        {
            actor_threads[i] = std::thread(&Training::actor, this);
        }
        std::thread learner_thread{&Training::learner, this};
        for (int i = 0; i < n_actor_threads; ++i)
        {
            actor_threads[i].join();
        }
        learner_thread.join();
    }
};

int main()
{
    const int sample_buffer_size = 1 << 12;
    const int learner_minibatch_size = 1 << 4;

    Training training_workspace{sample_buffer_size, learner_minibatch_size};
    training_workspace.train = false;
    // training_workspace.generate_samples = false;
    torch::load(training_workspace.net, "../saved/model_20231008144026.pt");
    // training_workspace.start(4);

    return 0;
}
