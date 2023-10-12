#include <torch/torch.h>

#include "./src/cuda.hh"

#include <queue>

#include "./sides.hh"

namespace Options
{
    const int batch_size = 1 << 10;
    const float policy_loss_weight = 0.5;
    const float base_learning_rate = .0001;
    const int samples_per_checkpoint = 1 << 20;
    const float log_learning_rate_decay = std::log(10) / -10;
    const float max_learn_actor_ratio = 150;
    const size_t full_iterations = 1 << 10;
    const size_t partial_iterations = 1 << 8;
    const float full_search_prob = .25;
    const int berth = 1 << 8;
    const size_t max_devices = 4;

    const torch::Device device0{torch::kCUDA, 0};
    const torch::Device device1{torch::kCUDA, 1};

    const int metric_history_size = 100;
};

#include "./src/nn.hh"
#include "./src/battle.hh"
#include "./src/cpu-model.hh"

void dummy_data(
    LearnerBuffers sample_buffers,
    const size_t count)
{
    BattleTypes::VectorReal strategy{9};
    strategy[0] = BattleTypes::Real{.5};
    strategy[1] = BattleTypes::Real{.5};
    std::vector<float> input{376};
    for (int b = 0; b < 376; ++b)
    {
        input[b] = static_cast<float>(b);
    }
    std::vector<int64_t> jpi{18};
    for (int b = 0; b < 18; ++b)
    {
        jpi[b] = b + 1;
    }
    for (size_t i = 0; i < count; ++i)
    {
        memcpy(&sample_buffers.float_input_buffer[376 * i], input.data(), 376 * sizeof(float));
        sample_buffers.value_data_buffer[i * 2] = .314;
        sample_buffers.value_data_buffer[i * 2 + 1] = .314;
        memcpy(
            &sample_buffers.joined_policy_index_buffer[i * 18],
            jpi.data(),
            18 * sizeof(int64_t));
        memcpy(
            &sample_buffers.joined_policy_buffer[i * 18],
            reinterpret_cast<float *>(strategy.data()),
            9 * 4);
        memcpy(
            &sample_buffers.joined_policy_buffer[i * 18 + 9],
            reinterpret_cast<float *>(strategy.data()),
            9 * 4);
    }
}

// Pinyon type list for the entire program: Single threaded search on battles using Monte-Carlo eval.
using Types = TreeBandit<Exp3<MonteCarloModel<BattleTypes>>, DefaultNodes>;
// using Types = FlatSearch<Exp3<MonteCarloModel<BattleTypes>>>;

// Metric to measure samples/sec that the actors generate and the learners consume
// the ratio between these is a basic parameter for this program
struct Metric
{
    std::mutex mtx{};
    const int max_donations;
    std::queue<int> donation_sizes;
    std::queue<decltype(std::chrono::high_resolution_clock::now())> donation_times;
    int total_donations = 0;

    Metric(const int max_donations) : max_donations{max_donations} {}

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

/*
shared_ptr type erasure is used because we need different underlying NN and optimizer types
*/

struct LearnerData
{
    const torch::Device device;
    const int batch_size;
    const float policy_loss_weight;
    const float base_learning_rate;
    const float log_learning_rate_decay;
    torch::nn::MSELoss mse{};
    torch::nn::CrossEntropyLoss cel{};
    std::vector<std::string> saves{};

    LearnerData(
        const torch::Device device = torch::kCPU,
        const int batch_size = Options::batch_size,
        const float base_learning_rate = Options::base_learning_rate,
        const float log_learning_rate_decay = Options::log_learning_rate_decay,
        const float policy_loss_weight = Options::policy_loss_weight)
        : device{device},
          batch_size{batch_size},
          base_learning_rate{base_learning_rate},
          log_learning_rate_decay{log_learning_rate_decay},
          policy_loss_weight{policy_loss_weight}
    {
    }

    Metric metric{100};
    uint64_t n_samples = 0;
    float learning_rate = base_learning_rate;
    // std::string get_string(const int) const { return ""; }
    virtual W::Types::Model make_w_model() const = 0;
    virtual void step(LearnerBuffers, bool) = 0;
};

template <typename Net, typename Optimizer>
struct LearnerImpl : LearnerData
{
    Net net;
    Optimizer optimizer;

    std::string get_string(const int checkpoint) const
    {
        return "";
    }

    template <typename... Args>
    LearnerImpl(
        const Net &net,
        const Args &...args)
        : LearnerData{args...},
          net{net},
          optimizer{net->parameters(), this->learning_rate}
    {
        this->net->to(this->device);
    }

    W::Types::Model make_w_model() const
    {
        // defines type list where `Model` is the output of exp3 search with the CPU net as the model
        using SearchModelTypes = TreeBanditSearchModel<TreeBandit<Exp3<CPUModel<Net>>>>;
        return W::make_model<SearchModelTypes>( //                          CPUNet params
            typename SearchModelTypes::Model{Options::full_iterations, {}, {""}, {.01}});
    }

    void step(
        LearnerBuffers learn_buffers,
        bool print = false) override
    {
        torch::Tensor float_input =
            torch::from_blob(
                learn_buffers.float_input_buffer,
                {this->batch_size, n_bytes_battle})
                .to(net->device);

        torch::Tensor value_data =
            torch::from_blob(
                learn_buffers.value_data_buffer,
                {this->batch_size, 2})
                .to(net->device);
        torch::Tensor q = value_data.index({"...", torch::indexing::Slice{0, 1, 1}});
        torch::Tensor z = value_data.index({"...", torch::indexing::Slice{1, 2, 1}});
        torch::Tensor value_target = .5 * q + .5 * z;

        torch::Tensor joined_policy_indices =
            torch::from_blob(
                learn_buffers.joined_policy_index_buffer,
                {this->batch_size, 18}, {torch::kInt64})
                .to(torch::kCUDA);
        torch::Tensor joined_policy_target =
            torch::from_blob(
                learn_buffers.joined_policy_buffer,
                {this->batch_size, 18})
                .to(net->device);
        torch::cuda::synchronize();

        torch::Tensor row_policy_target = joined_policy_target.index({"...", torch::indexing::Slice{0, 9, 1}});
        torch::Tensor col_policy_target = joined_policy_target.index({"...", torch::indexing::Slice{9, 18, 1}});
        row_policy_target /= row_policy_target.norm(1, {1}, true);
        col_policy_target /= col_policy_target.norm(1, {1}, true);

        NetOutput output = net->forward(float_input, joined_policy_indices);

        torch::Tensor value_loss =
            this->mse(output.value, value_target);
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

        if (print)
        {
            auto slice = torch::indexing::Slice(0, 4, 1);
            pt(output.value.index({slice, "..."}));
            pt(output.row_log_policy.index({slice, "..."}));
            pt(output.col_log_policy.index({slice, "..."}));
            std::cout << value_loss.item().toFloat();
        }

        torch::Tensor loss = value_loss + this->policy_loss_weight * (row_policy_loss + col_policy_loss);
        loss.backward();
        optimizer.step();
        optimizer.zero_grad();
    }
};

using Learner = std::shared_ptr<LearnerData>;

struct TrainingAndEval
{
    // ownership over metric (queue, mutex). `net` should be shared_ptr anyway
    std::vector<Learner> learners;
    // to keep track of training lifetime for 'divide-and-conquer` eval
    std::vector<std::vector<std::string>> learner_version_strings;
    // iterate over these to train on multiple GPUs at once
    std::vector<torch::Device> devices;
    std::vector<std::vector<Learner>> learner_device_groupings;

    PinnedBuffers sample_buffers;
    std::atomic<uint64_t> sample_index{0};
    // has the size of the largest learner `batch_size`
    DeviceBuffers
        learn_buffers_by_device[2];

    const int sample_buffer_size;
    const float max_learn_actor_ratio = Options::max_learn_actor_ratio;

    const size_t full_iterations = Options::full_iterations;
    const size_t partial_iterations = Options::partial_iterations;
    const float full_search_prob = Options::full_search_prob;
    const int berth = Options::berth;

    Metric actor_metric{Options::metric_history_size};
    Metric learn_metric{Options::metric_history_size};
    float actor_sample_rate;
    float learn_sample_rate; // total

    bool run_learn = false;
    bool run_actor = true;
    bool run_eval = false;

    TrainingAndEval(
        std::vector<Learner> &learners,
        const int sample_buffer_size)
        : learners{learners},
          sample_buffer_size{sample_buffer_size},
          sample_buffers{sample_buffer_size}
    {
        const size_t n_learners = learners.size();
        std::vector<size_t> max_batch_size_by_device = {};

        // Fill vector of devices and sort learners by device, also getting max sample size per device
        for (Learner learner : learners)
        {
            bool new_device = true;
            size_t device_index = 0;
            for (const auto &device : devices)
            {
                if (device == learner->device)
                {
                    new_device = false;
                    break;
                }
                ++device_index;
            }
            if (new_device)
            {
                devices.push_back(learner->device);
                learner_device_groupings.push_back({learner});
                max_batch_size_by_device.push_back(learner->batch_size);
            }
            else
            {
                learner_device_groupings[device_index].push_back(learner);
                if (learner->batch_size > max_batch_size_by_device[device_index])
                {
                    max_batch_size_by_device[device_index] = learner->batch_size;
                }
            }
        }

        // Initialize buffer for each device
        for (size_t device_index = 0; device_index < devices.size(); ++device_index)
        {
            // learn_buffers_by_device[device_index] = DeviceBuffers{max_batch_size_by_device[device_index]}; // TODO add device arg
        }
    }

    void actor_store(const PinnedActorBuffers &actor_buffers, const int count)
    {
        const uint64_t s = sample_index.fetch_add(count);
        const int sample_index_first = s % sample_buffer_size;

        actor_sample_rate = actor_metric.update_and_get_rate(count);
        if (actor_sample_rate > 0)
        {
            std::cout << "actor rate: " << actor_sample_rate << " count: " << count << std::endl;
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
            if (!run_learn)
            {
                run_learn = true;
                std::cout << "BUFFER FULL - TRAINING ENABLED" << std::endl;
            }
        }
    }

    void actor()
    {
        PinnedActorBuffers buffers{200};
        size_t buffer_index = 0;
        Types::PRNG device{};
        Types::State state{device.random_int(n_sides), device.random_int(n_sides)};
        Types::Model model{device.uniform_64()};
        Types::Search search{};
        int rows, cols;
        Types::Value value;
        Types::VectorReal row_strategy, col_strategy;
        while (run_actor)
        {
            if (state.is_terminal())
            {
                for (int i = 0; i < buffer_index; ++i)
                {
                    buffers.value_data_buffer[2 * i + 1] = state.payoff.get_row_value().get();
                }

                actor_store(buffers, buffer_index);

                buffer_index = 0;
                state = Types::State{device.random_int(n_sides), device.random_int(n_sides)};
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
                // copies the battle bytes and action indices
                state.copy_to_buffer(static_cast<ActorBuffers>(buffers), buffer_index, rows, cols);
                // search data now
                buffers.value_data_buffer[buffer_index * 2] = value.get_row_value().get();
                memcpy(
                    &buffers.joined_policy_buffer[buffer_index * 18],
                    reinterpret_cast<float *>(row_strategy.data()),
                    rows * 4);
                memcpy(
                    &buffers.joined_policy_buffer[buffer_index * 18 + 9],
                    reinterpret_cast<float *>(col_strategy.data()),
                    cols * 4);
                ++buffer_index;
            }
            state.apply_actions(state.row_actions[row_idx], state.col_actions[col_idx]);
            state.get_actions();
        }
    }

    void learn(const int device = 0)
    {
        while (!run_learn)
        {
            sleep(1);
        }

        size_t step = 0;
        while (run_learn)
        {
            
            // DeviceBuffers &learn_buffers = learn_buffers_by_device[device];
            switch_device(device);
            DeviceBuffers learn_buffers{2048, device};

            for (const Learner learner : learner_device_groupings[device])
            {
                const int start = (sample_index.load() + berth) % sample_buffer_size;
                // sample_mutex.lock();
                copy_sample_to_learn_buffer(
                    learn_buffers,
                    sample_buffers,
                    start,
                    sample_buffer_size - 2 * berth,
                    sample_buffer_size,
                    learner->batch_size);
                // sample_mutex.unlock();

                bool print = false;
                if (step % 100 == 0) {
                    std::cout << "device: " << device << std::endl;
                    print = true;
                }
                learner->step(learn_buffers, print);
            }
            ++step;
            // sleep(2);
        }
    }

    void eval()
    {
        while (!run_eval)
        {
            run_eval = false;
            sleep(1);
        }

        run_actor = false;
        run_learn = false;

        std::vector<W::Types::Model> agents{};
        for (const Learner learner : learners)
        {
            agents.emplace_back(learner->make_w_model());
        }
        using T = TreeBanditThreaded<Exp3<MonteCarloModel<Arena>>>;
        T::PRNG arena_device{};
        T::State arena_state{&battle_generator, agents};
        T::Model arena_model{0};
        T::Search arena_search{};
        T::MatrixNode root{};
        arena_search.run_for_iterations(1 << 10, arena_device, arena_state, arena_model, root);
    }

    static W::Types::State battle_generator(SimpleTypes::Seed seed)
    {
        SimpleTypes::PRNG device{seed};
        return W::make_state<BattleTypes>(device.random_int(n_sides), device.random_int(n_sides));
    }

    void run(
        const size_t n_actor_threads = 0)
    {
        const size_t n_devices = devices.size();
        std::thread actor_threads[n_actor_threads];
        std::thread learn_threads[Options::max_devices];

        for (int i = 0; i < n_actor_threads; ++i)
        {
            actor_threads[i] = std::thread(&TrainingAndEval::actor, this);
        }
        for (int i = 0; i < n_devices; ++i)
        {
            learn_threads[i] = std::thread(&TrainingAndEval::learn, this, i);
        }
        std::thread eval_thread{&TrainingAndEval::eval, this};

        for (int i = 0; i < n_actor_threads; ++i)
        {
            actor_threads[i].join();
        }
        for (int i = 0; i < n_devices; ++i)
        {
            learn_threads[i].join();
        }
        eval_thread.join();
    }
};

int main()
{
    auto l0 = std::make_shared<LearnerImpl<Net, torch::optim::SGD>>(
        Net{}, 
        Options::device0, 1024);
    auto l1 = std::make_shared<LearnerImpl<Net, torch::optim::SGD>>(
        Net{}, 
        Options::device1, 512);

    std::vector<Learner> learners = {l0, l1};

    const size_t sample_buffer_size = 1 << 10;
    TrainingAndEval workspace{learners, sample_buffer_size};
    dummy_data(workspace.sample_buffers, workspace.sample_buffer_size);
    workspace.run_actor = false;
    workspace.run_learn = true;
    workspace.run_eval = false;
    workspace.run();
    return 0;
}

// void learn()
// {
//     torch::nn::MSELoss mse{};
//     torch::nn::CrossEntropyLoss cel{};
//     float learning_rate = base_learning_rate;

//     while (!train)
//     {
//         sleep(1);
//     }

//     size_t checkpoint = 0;
//     while (checkpoint < 1000)
//     {
//         torch::optim::SGD optimizer{net->parameters(), learning_rate};
//         for (size_t step = 0; step < (1 << 7); ++step)
//         {

//             learn_fetch();

//             learn_rate = learn_metric.update_and_get_rate(learn_buffer_size);
//             while (learn_rate > max_learn_actor_ratio * actor_rate)
//             {
//                 sleep(1);
//                 learn_rate = learn_metric.update_and_get_rate(0);
//             }
//         }

//         if (checkpoint % 10 == 0)
//         {
//             auto timestamp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
//             std::ostringstream oss;
//             oss << std::put_time(std::gmtime(&timestamp), "%Y%m%d%H%M%S");
//             std::string timestampStr = oss.str();
//             std::string filename = "../saved/model_" + timestampStr + ".pt";
//             torch::save(net, filename);
//         }

//         ++checkpoint;
//         learning_rate *= std::exp(log_learning_rate_decay_per_checkpoint);
//     }
// }
