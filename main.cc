#include <torch/torch.h>

#include "./src/cuda.hh"

#include <queue>

#include "./sides.hh"

namespace Options
{
    const int batch_size = 1 << 10;
    const float policy_loss_weight = 0.5;
    const float base_learning_rate = .0001;
    const float log_learning_rate_decay = std::log(10) / -10;
    const float max_learn_actor_ratio = 150;
    const size_t full_iterations = 1 << 10;
    const size_t partial_iterations = 1 << 8;
    const size_t eval_iterations = 1 << 5;
    const float full_search_prob = .25;
    const int berth = 1 << 8;
    const size_t max_devices = 4; // only used as size for std::thread[]
    const int metric_history_size = 100;
    const float value_q_weight = .5;

    const size_t total_samples = 1 << (10 + 7);
    const size_t n_checkpoints = 2;
    const size_t n_samples_per_checkpoint = total_samples / n_checkpoints;
};

#include "./src/nn.hh"
#include "./src/battle.hh"
#include "./src/cpu-model.hh"
#include "./src/exp3-single.hh"

void dummy_data(
    LearnerBuffers sample_buffers,
    const size_t count)
{
    BattleTypes::VectorReal strategy{9};
    strategy[0] = BattleTypes::Real{.5};
    strategy[1] = BattleTypes::Real{.5};
    std::vector<float> input{};
    input.resize(376);
    for (int b = 0; b < 376; ++b)
    {
        input[b] = static_cast<float>(b);
    }
    std::vector<int64_t> jpi{};
    jpi.resize(18);
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
    float last_rate = 0;

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
        last_rate = 1000 * (total_donations - donation_sizes.front()) / duration;
        mtx.unlock();
        return last_rate;
    }
};

/*
shared_ptr type erasure is used because we need different underlying NN and optimizer types
*/

struct LearnerData
{
    const char cuda_device_index;
    const int batch_size;
    const float policy_loss_weight;
    const float base_learning_rate;
    const float log_learning_rate_decay;

    const torch::Device device{cuda_device_index == -1 ? torch::kCPU : torch::Device{torch::kCUDA, cuda_device_index}};
    torch::nn::MSELoss mse{};
    torch::nn::CrossEntropyLoss cel{};
    std::vector<std::string> saves{};
    Metric metric{Options::metric_history_size};
    const size_t max_samples = Options::total_samples;
    const size_t n_steps_to_completion = max_samples / batch_size;
    size_t n_samples = 0;
    float learning_rate = base_learning_rate;

    LearnerData(
        const char cuda_device_index = 0,
        const int batch_size = Options::batch_size,
        const float base_learning_rate = Options::base_learning_rate,
        const float log_learning_rate_decay = Options::log_learning_rate_decay,
        const float policy_loss_weight = Options::policy_loss_weight)
        : cuda_device_index{cuda_device_index},
          batch_size{batch_size},
          policy_loss_weight{policy_loss_weight},
          base_learning_rate{base_learning_rate},
          log_learning_rate_decay{log_learning_rate_decay}
    {
    }

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
        if (this->cuda_device_index != char{-1})
        {
            this->net->to(this->device);
        }
    }

    W::Types::Model make_w_model() const
    {
        // defines type list where `Model` is the output of exp3 search with the CPU net as the model
        using SearchModelTypes = TreeBanditSearchModel<TreeBandit<Exp3<CPUModel<Net>>>>;
        return W::make_model<SearchModelTypes>( //                          CPUNet params
            typename SearchModelTypes::Model{Options::eval_iterations, {}, {net}, {.01}});
    }

    void step(
        LearnerBuffers learn_buffers,
        bool print = false) override
    {
        torch::Tensor float_input =
            torch::from_blob(
                learn_buffers.float_input_buffer,
                {this->batch_size, n_bytes_battle})
                .to(this->device);

        torch::Tensor value_data =
            torch::from_blob(
                learn_buffers.value_data_buffer,
                {this->batch_size, 2})
                .to(this->device);
        torch::Tensor q = value_data.index({"...", torch::indexing::Slice{0, 1, 1}});
        torch::Tensor z = value_data.index({"...", torch::indexing::Slice{1, 2, 1}});
        torch::Tensor value_target = Options::value_q_weight * q + (1 - Options::value_q_weight) * z;

        torch::Tensor joined_policy_indices =
            torch::from_blob(
                learn_buffers.joined_policy_index_buffer,
                {this->batch_size, 18}, {torch::kInt64})
                .to(this->device);
        torch::Tensor joined_policy_target =
            torch::from_blob(
                learn_buffers.joined_policy_buffer,
                {this->batch_size, 18})
                .to(this->device);
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
        }

        torch::Tensor loss = value_loss + this->policy_loss_weight * (row_policy_loss + col_policy_loss);
        loss.backward();
        optimizer.step();
        optimizer.zero_grad();

        const size_t old_checkpoint = n_samples / Options::n_samples_per_checkpoint;
        n_samples += batch_size;
        const size_t new_checkpoint = n_samples / Options::n_samples_per_checkpoint;

        if (old_checkpoint != new_checkpoint)
        {
            std::cout << "CHECKPOINT " << new_checkpoint << " REACHED" << std::endl;
        }
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
    std::vector<char> device_indices;
    // std::vector<std::vector<Learner>> learner_device_groupings;
    std::unordered_map<char, std::vector<Learner>> learner_device_groupings;
    // has the size of the largest learner `batch_size`

    std::unordered_map<char, LearnerBuffers>
        raw_learn_buffers;
    std::unordered_map<char, int>
        device_batch_size{};

    LearnerBuffers sample_buffers{};
    std::atomic<uint64_t> sample_index{0};

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
          sample_buffer_size{sample_buffer_size}
    {
        std::cout << "Init TrainingAndEval" << std::endl;
        const size_t n_learners = learners.size();
        std::cout << n_learners << " learners" << std::endl;

        // Fill vector of devices and sort learners by device, also getting max sample size per device
        for (const Learner &learner : learners)
        {
            const auto iter = device_batch_size.find(learner->cuda_device_index);
            const bool new_device = (iter == device_batch_size.end());

            if (new_device)
            {
                std::cout << "New device " << (int)learner->cuda_device_index << " batch size: " << learner->batch_size << std::endl;
                const char x = learner->cuda_device_index;
                device_indices.emplace_back(x);
                learner_device_groupings[learner->cuda_device_index] = {learner};
                device_batch_size[learner->cuda_device_index] = learner->batch_size;
            }
            else
            {
                std::cout << "Old device: " << (int)learner->cuda_device_index << " batch size: " << learner->batch_size << std::endl;
                learner_device_groupings[learner->cuda_device_index].push_back(learner);
                if (learner->batch_size > device_batch_size[learner->cuda_device_index])
                {
                    device_batch_size[learner->cuda_device_index] = learner->batch_size;
                }
            }
        }

        for (const auto pair : device_batch_size)
        {
            assert(pair.first != char{-1});
            switch_device(pair.first);
            alloc_device_buffers(raw_learn_buffers[pair.first], pair.second);
        }
        alloc_pinned_buffers(sample_buffers, sample_buffer_size);
    }

    ~TrainingAndEval()
    {
        for (auto pair : raw_learn_buffers)
        {
            switch_device(pair.first);
            dealloc_buffers(pair.second);
        }
        dealloc_buffers(sample_buffers);
    }

    void actor_store(const ActorBuffers &actor_buffers, const int count)
    {
        const uint64_t s = sample_index.fetch_add(count);
        const int sample_index_first = s % sample_buffer_size;

        actor_sample_rate = actor_metric.update_and_get_rate(count);
        if (actor_sample_rate > 0)
        {
            std::cout << "actor rate: " << actor_sample_rate << " count: " << count << std::endl;
        }

        copy_game_to_sample_buffer(
            sample_buffers,
            actor_buffers,
            sample_index_first,
            count,
            sample_buffer_size);

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
        ActorBuffers buffers{};
        alloc_actor_buffers(buffers, 200);
        // resets every game, basically equivalent to turns sampled
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
        dealloc_actor_buffers(buffers);
    }

    void learn(const char device_index = 0)
    {
        switch_device(device_index);
        Metric device_metric{Options::metric_history_size};

        while (!run_learn)
        {
            sleep(1);
        }

        std::cout << "START LEARN ON DEVICE: " << (int)device_index << std::endl;
        size_t step = 0;
        float device_rate = 0;

        while (run_learn)
        {
            LearnerBuffers learn_buffers = raw_learn_buffers[device_index];
            std::vector<Learner> &device_learners = learner_device_groupings[device_index];

            // erases from learner_device_groupings but not this->learners, so its not really deleted
            int learner_index = 0;
            auto n_finished = std::erase_if(
                device_learners,
                [&learner_index, &device_index](const Learner &learner)
                {
                    const bool finished = learner->n_samples >= learner->max_samples;
                    if (finished)
                    {
                        std::cout << "LEARNER " << learner_index << " FINISHED ON DEVICE: " << (int)device_index << std::endl;
                    }
                    ++learner_index;
                    return finished;
                });

            // check if terminate
            const size_t n_learners = device_learners.size();
            if (n_learners == 0)
            {
                std::cout << "LEARNING ON DEVICE " << (int)device_index << " FININSHED" << std::endl;
                break;
            }

            // now step with remaining learners
            for (const Learner &learner : device_learners)
            {

                // larger batch sizes get skipped sometimes so smaller batch sizes can keep up
                (void)learner->metric.update_and_get_rate(0);
                if (learner->metric.last_rate * n_learners > device_rate)
                {
                    continue;
                }

                const int start = (sample_index.load() + berth) % sample_buffer_size;
                copy_sample_to_learn_buffer(
                    learn_buffers,
                    sample_buffers,
                    start,
                    sample_buffer_size - 2 * berth,
                    sample_buffer_size,
                    learner->batch_size);

                bool print = false;
                if (step % 1000 == 0)
                {
                    std::cout << "learner: " << learner.get() << "; device index: " << (int)device_index << std::endl;
                    std::cout << device_rate << " samples/sec (device)" << std::endl;
                    print = true;
                }
                // increments learner->n_samples
                learner->step(learn_buffers, print);

                device_rate = device_metric.update_and_get_rate(learner->batch_size);
                (void)learner->metric.update_and_get_rate(learner->batch_size);
            }

            device_rate = device_metric.update_and_get_rate(0);
            if (device_rate > n_learners * actor_sample_rate * Options::max_learn_actor_ratio)
            {
                using namespace std::chrono_literals;
                std::this_thread::sleep_for(50ms);
            }

            ++step;
        }
        std::cout << "LEARNER END, STEPS: " << step << std::endl;
    }

    static W::Types::State battle_generator(SimpleTypes::Seed seed)
    {
        SimpleTypes::PRNG device{seed};
        return W::make_state<BattleTypes>(device.random_int(n_sides), device.random_int(n_sides));
    }

    void eval()
    {
        const size_t n_threads = 6;
        const size_t n_iter = 1 << 7;
        const size_t n_prints = 1 << 3;
        const size_t n_iter_per_print = n_iter / n_prints;

        while (!run_eval)
        {
            sleep(1);
        }

        run_actor = false;
        run_learn = false;

        std::cout << "STARTING EVAL" << std::endl;

        // monte carlo search model as a control
        using _MCTSTypes = TreeBanditSearchModel<TreeBandit<Exp3<MonteCarloModel<BattleTypes>>>>;

        for (const Learner &learner : learners)
        {
            // get best agent for all w/e
        }

        std::vector<W::Types::Model>
            agents{// iter, device, model, search
                   //    W::make_model<_MCTSTypes>(_MCTSTypes::Model{1 << 8, {}, {0}, {}}),
                   W::make_model<_MCTSTypes>(_MCTSTypes::Model{1 << 10, {}, {0}, {}})};
        //    W::make_model<_MCTSTypes>(_MCTSTypes::Model{1 << 12, {}, {0}, {}})};

        // add the CPU versions of the learner nets
        for (const Learner learner : learners)
        {
            agents.emplace_back(learner->make_w_model());
        }

        // Exp3single is for arena, i.e. symmetric one shot games
        using A = TreeBanditThreaded<Exp3Single<MonteCarloModel<Arena>>>;
        A::PRNG arena_device{};
        A::State arena_state{&battle_generator, agents};
        A::Model arena_model{0};
        A::Search arena_search{{.01}, n_threads};
        A::MatrixNode root{};

        // init on single thread, kinda jank, remove?
        A::ModelOutput output{};
        A::State arena_state_{arena_state};
        arena_state_.randomize_transition(arena_device);
        arena_search.run_iteration(arena_device, arena_state_, arena_model, &root, output);

        for (size_t print = 0; print < n_prints; ++print)
        {
            arena_search.run_for_iterations(n_iter_per_print, arena_device, arena_state, arena_model, root);
            std::cout << "EVAL UPDATE " << print << "/" << n_prints << std::endl;
            std::cout << "CUMULATIVE VALUES:" << std::endl;
            root.stats.cum_values.print();
            std::cout << "VISITS:" << std::endl;
            root.stats.joint_visits.print();
        }
    }

    // simply start all threads. each function handles its own start/stop
    void run(
        const size_t n_actor_threads = 0)
    {
        const size_t n_devices = device_indices.size();
        std::thread actor_threads[n_actor_threads];
        std::thread learn_threads[Options::max_devices];

        for (size_t i = 0; i < n_actor_threads; ++i)
        {
            actor_threads[i] = std::thread(&TrainingAndEval::actor, this);
        }
        // for (size_t i = 0; i < n_devices; ++i)
        // {
        //     learn_threads[i] = std::thread(&TrainingAndEval::learn, this, device_indices[i]);
        // }
        // for (size_t i = 0; i < n_devices; ++i)
        // {
        //     learn_threads[i].join();
        // }
        run_actor = false;
        run_learn = false;
        run_eval = true;
        for (size_t i = 0; i < n_actor_threads; ++i)
        {
            actor_threads[i].join();
        }

        std::thread eval_thread{&TrainingAndEval::eval, this};
        eval_thread.join();
    }
};

int main()
{
    auto l0 = std::make_shared<LearnerImpl<FCResNet, torch::optim::SGD>>(
        FCResNet{}, char{0}, // Net, CUDA device index
        1024);               // optional hyperparams e.g. batch size etc.
    auto l1 = std::make_shared<LearnerImpl<FCResNet, torch::optim::SGD>>(
        FCResNet{}, char{0},
        1024);
    auto l2 = std::make_shared<LearnerImpl<FCResNet, torch::optim::SGD>>(
        FCResNet{}, char{0},
        1024);
    auto k0 = std::make_shared<LearnerImpl<FCResNet, torch::optim::SGD>>(
        FCResNet{}, char{1},
        1024);

    // in my benchmarking, my 2nd GPU is 3x slower. Thus the first GPU has 3 times as many nets to train
    std::vector<Learner> learners = {l0, l1, l2, k0};

    const size_t sample_buffer_size = 1 << 10;
    TrainingAndEval workspace{learners, sample_buffer_size};
    dummy_data(workspace.sample_buffers, workspace.sample_buffer_size);
    // workspace.run_actor = true;
    // workspace.run_learn = true;
    // workspace.run_eval = false;
    workspace.run_actor = false;
    workspace.run_learn = false;
    workspace.run_eval = true;

    const size_t n_actor_threads = 4;
    workspace.run(n_actor_threads);
    return 0;
}
