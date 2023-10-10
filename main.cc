#include <torch/torch.h>

#include <pinyon.hh>
#include <pkmn.h>
#include "./src/cuda.hh"

#include <queue>

#include "./sides.hh"

namespace Options
{
    const int batch_size = 1 << 10;
    const float policy_loss_weight = 0.5;
    const float base_learning_rate = .01;
    const int samples_per_checkpoint = 1 << 20;
    const float log_learning_rate_decay_per_checkpoint = std::log(10) / -10;
    const float max_learn_actor_ratio = 150;
    const size_t full_iterations = 1 << 10;
    const size_t partial_iterations = 1 << 8;
    const float full_search_prob = .25;
    const int berth = 1 << 8;

    const int metric_history_size = 100;
};

#include "./src/res-net.hh"
#include "./src/battle.hh"

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

// Pinyon type list for the entire program: Single threaded search on battles using Monte-Carlo eval.
using Types = TreeBandit<Exp3<MonteCarloModel<BattleTypes>>, DefaultNodes>;
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

struct TraineeData
{
    torch::Device device;
    const int batch_size;
    const float policy_loss_weight;
    const float base_learning_rate;
    const float log_learning_rate_decay_per_checkpoint;

    TraineeData(
        const torch::Device device = torch::kCUDA,
        const int batch_size = Options::batch_size,
        const float policy_loss_weight = Options::policy_loss_weight,
        const float base_learning_rate = Options::base_learning_rate,
        const float log_learning_rate_decay_per_checkpoint = Options::log_learning_rate_decay_per_checkpoint)
        : device{device},
          batch_size{batch_size},
          policy_loss_weight{policy_loss_weight},
          base_learning_rate{base_learning_rate},
          log_learning_rate_decay_per_checkpoint{log_learning_rate_decay_per_checkpoint}
    {
    }

    Metric metric{100};
    uint64_t n_samples = 0;
    float learning_rate = base_learning_rate;

    virtual std::string get_string(const int) const = 0;
    torch::Device get_device() const { return device; }
    int get_batch_size() const { return batch_size; }
    // std::string get_string(const int) const { return ""; }
};

template <typename Net, typename Optimizer>
struct TraineeImpl : TraineeData
{
    // torch::nn::Module net{nullptr};
    Net net;
    Optimizer optimizer;

    std::string get_string(const int checkpoint) const
    {
        return "";
    }

    template <typename... Args>
    TraineeImpl(
        const Net &net,
        const Args &...args)
        : TraineeData{args...},
          net{net},
          optimizer{net->parameters(), this->learning_rate}
    {
        // optimizer = std::move(Optimizer{net->parameters(), base_learning_rate});
    }
};

using Trainee = std::shared_ptr<TraineeData>;

struct TrainingAndEval
{
    // ownership over metric (queue, mutex). `net` should be shared_ptr anyway
    std::vector<Trainee> trainees;
    // to keep track of training lifetime for 'divide-and-conquer` eval
    std::vector<std::vector<std::string>> trainee_version_strings;
    // iterate over these to train on multiple GPUs at once
    std::vector<torch::Device> training_devices;
    std::vector<std::vector<Trainee>> trainee_device_groupings;

    PinnedBuffers sample_buffers{
        sample_buffer_size};
    std::atomic<uint64_t> sample_index{0};
    // has the size of the largest learner `batch_size`
    std::vector<
        DeviceBuffers>
        learn_buffers_by_device;

    const int sample_buffer_size;
    const int learn_buffer_size;
    const float max_learn_actor_ratio = Options::max_learn_actor_ratio;

    const size_t full_iterations = Options::full_iterations;
    const size_t partial_iterations = Options::partial_iterations;
    const float full_search_prob = Options::full_search_prob;
    const int berth = Options::berth;

    Metric actor_metric{Options::metric_history_size};
    Metric learn_metric{Options::metric_history_size};
    float actor_sample_rate;
    float learn_sample_rate; // average

    bool run_learn = false;
    bool run_actor = true;

    template <typename... TraineeTs>
    TrainingAndEval(
        const TraineeTs &&...trainee_args,
        const int sample_buffer_size)
        : // : trainees{std::forward(trainee_args...)},
          sample_buffer_size{sample_buffer_size}
    {
        trainees = {std::make_shared<TraineeTs>(std::forward(trainee_args...))...};
        const size_t n_trainees = trainees.size();
        std::vector<size_t> max_batch_size_by_device = {};

        for (auto trainee : trainees)
        {
            bool assumed_new_device = true;
            size_t device_index = 0;
            for (const auto &device : training_devices)
            {
                if (device == trainee->get_device())
                {
                    assumed_new_device = false;
                    break;
                }
                ++device_index;
            }
            if (assumed_new_device)
            {
                training_devices.push_back(trainee->get_device());
                trainee_device_groupings.emplace_back(trainee);
                max_batch_size_by_device.push_back(trainee->get_batch_size());
            }
            else
            {
                trainee_device_groupings[device_index].push_back(trainee);
                if (trainee->get_batch_size() > max_batch_size_by_device[device_index])
                {
                    max_batch_size_by_device[device_index] = trainee->get_batch_size();
                }
            }
        }

        for (size_t device_index = 0; device_index < training_devices.size(); ++device_index)
        {
            learn_buffers_by_device.emplace_back(max_batch_size_by_device[device_index]); // TODO add device arg
        }
    }

    void actor() {}
    void learn()
    {

        while (!run_learn)
        {
            sleep(1);
        }

        while (run_learn)
        {
        }
    }
    void run() {}
    void train(
        const size_t n_actor_threads)
    {
        std::thread actor_threads[n_actor_threads];
        for (int i = 0; i < n_actor_threads; ++i)
        {
            actor_threads[i] = std::thread(&TrainingAndEval::actor, this);
        }
        std::thread learn_thread{&TrainingAndEval::learn, this};
        for (int i = 0; i < n_actor_threads; ++i)
        {
            actor_threads[i].join();
        }
        learn_thread.join();
    }
    void eval() {}
};

struct Training
{
    Net net;
    bool train = true;
    bool generate_samples = true;

    const int sample_buffer_size;
    const int learn_buffer_size;
    const float policy_loss_weight = 1.0f;
    const float base_learning_rate = .001;
    const float log_learning_rate_decay_per_checkpoint = std::log(10) / -1000;
    const float max_learn_actor_ratio = 150;

    PinnedBuffers sample_buffers{
        sample_buffer_size};
    std::atomic<uint64_t> sample_index{0};

    DeviceBuffers learn_buffers{
        learn_buffer_size};

    PinnedBuffers index_buffers{
        learn_buffer_size};

    const size_t full_iterations = 1 << 10;
    const size_t partial_iterations = 1 << 8;
    const float full_search_prob = .25;
    const int berth = 1 << 8;

    std::mutex sample_mutex{};

    Metric actor_metric{100};
    Metric learn_metric{100};
    float actor_rate;
    float learn_rate;

    template <typename... Args>
    Training(
        Args... args,
        const int sample_buffer_size,
        const int learn_buffer_size)
        : net{args...},
          sample_buffer_size{sample_buffer_size},
          learn_buffer_size{learn_buffer_size}
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

    void learn_fetch()
    {
        const int start = (sample_index.load() + berth) % sample_buffer_size;
        // sample_mutex.lock();
        copy_sample_to_learn_buffer(
            learn_buffers,
            sample_buffers,
            index_buffers,
            start,
            sample_buffer_size - 2 * berth,
            sample_buffer_size,
            learn_buffer_size);
        // sample_mutex.unlock();
    };

    void actor()
    {
        PinnedActorBuffers buffers{500};
        int buffer_index = 0;
        Types::PRNG device{};
        Types::State state{device.random_int(n_sides), device.random_int(n_sides)};
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
            const int row_idx = device.sample_pdf(row_strategy);
            const int col_idx = device.sample_pdf(col_strategy);
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

    void learn()
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

                learn_fetch();

                torch::Tensor float_input =
                    torch::from_blob(
                        learn_buffers.float_input_buffer,
                        {learn_buffer_size, n_bytes_battle})
                        .to(net->device);

                torch::Tensor value_data =
                    torch::from_blob(
                        learn_buffers.value_data_buffer,
                        {learn_buffer_size, 2})
                        .to(net->device);
                torch::Tensor q = value_data.index({"...", torch::indexing::Slice{0, 1, 1}});
                torch::Tensor z = value_data.index({"...", torch::indexing::Slice{1, 2, 1}});
                torch::Tensor value_target = .5 * q + .5 * z;

                torch::Tensor joined_policy_indices =
                    torch::from_blob(
                        learn_buffers.joined_policy_index_buffer,
                        {learn_buffer_size, 18}, {torch::kInt64})
                        .to(torch::kCUDA);
                torch::Tensor joined_policy_target =
                    torch::from_blob(
                        learn_buffers.joined_policy_buffer,
                        {learn_buffer_size, 18})
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
                        output.row_log_policy,
                        row_policy_target,
                        torch::nn::functional::KLDivFuncOptions(torch::kBatchMean));
                torch::Tensor col_policy_loss =
                    torch::nn::functional::kl_div(
                        output.col_log_policy,
                        col_policy_target,
                        torch::nn::functional::KLDivFuncOptions(torch::kBatchMean));

                if (step == 0)
                {
                    std::cout << "value target (first 4 entries)" << std::endl;
                    pt(value_target.index({torch::indexing::Slice(0, 4, 1)}));
                    std::cout << "value output (first 4 entries)" << std::endl;
                    pt(output.value.index({torch::indexing::Slice(0, 4, 1)}));
                    std::cout << "value loss (batch): " << value_loss.item().toFloat() << std::endl;
                }

                torch::Tensor loss = value_loss + policy_loss_weight * (row_policy_loss + col_policy_loss);
                loss.backward();
                optimizer.step();
                optimizer.zero_grad();

                learn_rate = learn_metric.update_and_get_rate(learn_buffer_size);
                while (learn_rate > max_learn_actor_ratio * actor_rate)
                {
                    sleep(1);
                    learn_rate = learn_metric.update_and_get_rate(0);
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
            learning_rate *= std::exp(log_learning_rate_decay_per_checkpoint);
        }
    }


};

void count_transitions(
    const Types::State &state,
    const int tries = 1000,
    const int row_idx = 0)
{
    Types::PRNG device{};
    Types::ObsHash hasher{};
    std::unordered_map<uint64_t, int> count{};

    std::cout << "row idx" << row_idx << std::endl;
    for (int i = 0; i < tries; ++i)
    {
        Types::State state_ = state;
        state_.randomize_transition(device);
        state_.apply_actions(
            state_.row_actions[row_idx],
            state_.col_actions[row_idx]);
        // 14,14);

        // for (int j = 0; j < 64; ++j)
        // {
        //     std::cout << static_cast<int>(state_.obs.get()[j]) << " ";
        // }
        // std::cout << std::endl;

        uint64_t hash = hasher(state_.obs);
        count[hash] += 1;
    }

    for (const auto &x : count)
    {
        std::cout << x.first << " : " << x.second << std::endl;
    }
}

void branch()
{
    Types::State state{0, 0};
    state.apply_actions(0, 0);
    state.get_actions();

    for (int i = 0; i < 9; ++i)
    {
        count_transitions(state, 1000, i);
        std::cout << std::endl;
    }
}

int main()
{
    TraineeImpl<Net, torch::optim::SGD>(Net{}, torch::kCUDA);
    // TrainingAndEval workstation{
    //     Net{},
    //     1 << 16};

    // const int sample_buffer_size = 1 << 12;
    // const int learn_minibatch_size = 1 << 10;

    // Training training_workspace{sample_buffer_size, learn_minibatch_size};
    // training_workspace.train = false;
    // training_workspace.start(4);

    // using NewTypes = TreeBandit<Exp3<NNTypes>>;

    // NewTypes::Model model{};
    // NewTypes::State battle{0};
    // battle.apply_actions(0, 0);
    // battle.get_actions();

    // NewTypes::Search search{};
    // NewTypes::PRNG device{};
    // NewTypes::MatrixNode root{};
    // size_t ms = search.run_for_iterations(1000, device, battle, model, root);
    // std::cout << ms << " ms for 1000 iter" << std::endl;

    // NewTypes::VectorReal row_strategy, col_strategy;
    // NewTypes::Value value;
    // search.get_empirical_strategies(root.stats, row_strategy, col_strategy);
    // search.get_empirical_value(root.stats, value);

    // math::print(row_strategy);
    // math::print(col_strategy);
    // std::cout << value << std::endl;

    return 0;
}