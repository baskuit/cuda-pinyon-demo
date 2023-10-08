#include <torch/torch.h>

#include <pinyon.hh>
#include <pkmn.h>

#include "./src/cuda.hh"

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
    const float policy_loss_weight = 1.0f;
    const float base_learning_rate = .001;
    const float log_learning_rate_decay_per_step = std::log(10) / -1000;
    const float max_learner_actor_ratio = 150;

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
    float actor_rate;
    float learner_rate;

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
    }

    Training(const Training &) = delete;

    void actor_store(const PinnedActorBuffers &actor_buffers, const int count)
    {
        const uint64_t s = sample_index.fetch_add(count);
        const int sample_index_first = s % sample_buffer_size;

        actor_rate = actor_metric.update_and_get_rate(count);
        if (actor_rate > 0)
        {
            std::cout << "actor rate: " << actor_rate << " count: " << count << std::endl;
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
        torch::nn::MSELoss mse{};
        torch::nn::CrossEntropyLoss cel{};
        float learning_rate = base_learning_rate;

        while (!train)
        {
            sleep(1);
        }

        size_t checkpoint = 0;
        while (checkpoint < 1000)
        {
            torch::optim::SGD optimizer{net->parameters(), learning_rate};
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
                torch::Tensor q = value_data.index({"...", torch::indexing::Slice{0, 1, 1}});
                torch::Tensor z = value_data.index({"...", torch::indexing::Slice{1, 2, 1}});
                torch::Tensor value_target = .5 * q + .5 * z;

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

                torch::Tensor loss = value_loss + policy_loss_weight * (row_policy_loss + col_policy_loss);
                loss.backward();
                optimizer.step();
                optimizer.zero_grad();
                // pt(row_policy_target.index({torch::indexing::Slice(0, 4, 1), "..."}));
                // pt(output.row_policy.index({torch::indexing::Slice(0, 4, 1), "..."}));

                learner_rate = learn_metric.update_and_get_rate(learner_buffer_size);
                while (learner_rate > max_learner_actor_ratio * actor_rate)
                {
                    sleep(1);
                    learner_rate = learn_metric.update_and_get_rate(0);
                }
            }

            if (checkpoint % 10 == 0)
            {
                auto timestamp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                std::ostringstream oss;
                oss << std::put_time(std::gmtime(&timestamp), "%Y%m%d%H%M%S");
                std::string timestampStr = oss.str();
                std::string filename = "../saved/model_" + timestampStr + ".pt";
                torch::save(net, filename);
            }

            ++checkpoint;
            learning_rate *= std::exp(log_learning_rate_decay_per_step);
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
    const int sample_buffer_size = 1 << 16;
    const int learner_minibatch_size = 1 << 10;

    Training training_workspace{sample_buffer_size, learner_minibatch_size};
    training_workspace.train = false;
    // training_workspace.generate_samples = false;
    // torch::load(training_workspace.net, "../saved/model_20231008144026.pt");
    training_workspace.start(4);

    return 0;
}
