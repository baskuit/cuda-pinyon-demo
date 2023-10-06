#include <pinyon.hh>
#include <pkmn.h>

#include <queue>

#include "./src/battle.hh"
#include "./src/common.hh"
#include "./src/buffers.hh"
#include "./src/search.hh"

#ifdef ENABLE_TORCH
#include <torch/torch.h>
#include "./src/net.hh"
#else
struct Net
{
};
#endif

// Pinyon type list for the entire program: Single threaded search on battles using Monte-Carlo eval.
using Types = TreeBandit<Exp3<MonteCarloModel<BattleTypes>>, FlatNodes>;
// using Types = FlatSearch<Exp3<MonteCarloModel<BattleTypes>>>;

// Need Types defined first
// #include "./src/scripts.hh"

/*
Initializses buffers, contains synch mechanism for actor and learner threads

This data will be used by >= 1 actor threads and 1 learner thread.
Actors do self-play training games and write their raw battle data to the a pinned 'producer buffer'
They simply write input, value, policy, etc tensors at an atomicly fetched and incremented index
    that wraps around when it overflows the buffer bounds
The program exploits the pinning with a CUDA kernel that converts the `uint64_t[47]` battle data into `float[n_bytes_battle]` data

The producer buffer and the learner buffer are split into uniform 'blocks' (of 'block_size' many samples) for transfer.
(Each sample has byte size which is the sum of the value, policy, etc byte sizes.
For example the raw_input part of a single sample is n_bytes_battle bytes, and its float version is n_bytes_battle * 4 bytes.)
When a thread is assigned the buffer index at the end of a block, after writing to the production buffer it will
store a block into the consumer buffer. The stored block is the *previous* block (wrapping around) to the one it just finished writing to
(this is a soft-check to slower, lower index threads writing to the chunck as its being copied. very unlikely with large enough block size.)

The rate of storage is relatively low since a full search needs to be performed for every sample,
so the learner thread is able to do its work much faster.

The learner thread has its own buffer for storing all the training games. This demo does not use the disk,
so this is a large CPU buffer defined with std::vectors
It samples learner minibatches from this buffer constantly and trains a libtorch model.
There is a (wrapping) range of blocks that are not in any danger of being written to during one
retrieval of a minibatch, given by the current consumer block index, because of the speed disparity.
*/

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

    const size_t full_iterations = 1 << 10;
    const size_t partial_iterations = 1 << 8;
    const float full_search_prob = .25;

    std::mutex sample_mutex{};

    // simple class to count the samples/sec for both actors and learner.
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
        // #ifdef ENABLE_TORCH
        net.to(torch::kCUDA);
        for (int i = 0; i < sample_buffer_size; ++i)
        {
            for (int j = 0; j < 18; ++j)
            {
                sample_buffers.joined_policy_index_buffer[i * 18 + j] = int64_t{i % 100};
            }
            // memset(&sample_buffers.joined_policy_index_buffer[18 * i], i, 18 * sizeof(int64_t));
        }
        // torch::Tensor sample_buffer_index_tensor =
        //     torch::from_blob(
        //         sample_buffers.joined_policy_index_buffer,
        //         {sample_buffer_size, 18}, {torch::kInt64});
        // std::cout << "entire sample " << std::endl;
        // pt(sample_buffer_index_tensor);
        // #endif
    }

    Training(const Training &) = delete;

    const int berth = 1 << 6;

    void actor_store(const PinnedActorBuffers &actor_buffers, const int count)
    {
        const uint64_t s = sample_index.fetch_add(count);
        const int sample_index_first = s % sample_buffer_size;

        float rate = actor_metric.update_and_get_rate(count);
        if (rate > 0)
        {
            std::cout << "actor rate: " << rate << " count: " << count << std::endl;
        }

        sample_mutex.lock();
        CUDACommon::copy_game_to_sample_buffer(
            sample_buffers,
            actor_buffers,
            sample_index_first,
            count,
            sample_buffer_size);
        sample_mutex.unlock();

        // std::cout << "store from;for " << sample_index_first << ";" << count << std::endl;

        if (s > sample_buffer_size)
        {
            if (!train)
            {
                train = true;
                std::cout << "TRAINING ENABLED" << std::endl;
                // generate_samples = false;
            }
        }
    }

    void learner_fetch()
    {
        const int center = (sample_index.load()) % sample_buffer_size;
        const int low = (center + berth) % sample_buffer_size;
        const int high = (center + sample_buffer_size - berth) % sample_buffer_size;
        int range;
        if (low < high)
        {
            range = high - low;
        }
        else
        {
            range = sample_buffer_size - high + low;
        }
        // std::cout << low << ' ' << center << ' ' << high << std::endl;
        sample_mutex.lock();

        CUDACommon::copy_sample_to_learner_buffer(
            learner_buffers,
            sample_buffers,
            index_buffers,
            low,
            range,
            sample_buffer_size,
            learner_buffer_size);
        sample_mutex.unlock();
    };

    void actor()
    {
        // most samples possible per game
        const int max_samples = 500;
        // current index on actor buffer
        PinnedActorBuffers buffers{max_samples};
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
                // battle
                memcpy(
                    &buffers.raw_input_buffer[buffer_index * 47],
                    reinterpret_cast<const uint64_t *>(state.battle.bytes),
                    376);
                // value
                buffers.value_data_buffer[buffer_index * 2] = value.get_row_value().get();
                // policy
                memcpy(
                    &buffers.joined_policy_buffer[buffer_index * 18],
                    reinterpret_cast<float *>(row_strategy.data()),
                    rows * 4);
                memcpy(
                    &buffers.joined_policy_buffer[buffer_index * 18 + 9],
                    reinterpret_cast<float *>(col_strategy.data()),
                    cols * 4);
                // getting indices
                auto get_index_from_action = [](const uint8_t *bytes, const uint8_t choice, const uint8_t col_offset = 0)
                {
                    uint8_t type = choice & 3;
                    uint8_t data = choice >> 2;
                    if (type == 1)
                    {
                        uint8_t moveid = bytes[2 * (data - 1) + 10 + col_offset];
                        return int64_t{moveid};
                    }
                    else if (type == 2)
                    {
                        uint8_t slot = bytes[176 + data - 1 + col_offset];
                        int dex = bytes[24 * (slot - 1) + 21 + col_offset];
                        return int64_t{dex + 165};
                    }
                    else
                    {
                        return int64_t{666}; // TODO fix what is this?
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
                        buffers.joined_policy_index_buffer[buffer_index * 18 + i] = 77; // TODO change to real dummy
                    }
                }
                for (int i = 0; i < 9; ++i)
                {
                    if (i < cols)
                    {
                        buffers.joined_policy_index_buffer[buffer_index * 18 + 9 + i] = get_index_from_action(state.battle.bytes, state.col_actions[i].get(), 184);
                    }
                    else
                    {
                        buffers.joined_policy_index_buffer[buffer_index * 18 + 9 + i] = 77;
                    }
                }

                ++buffer_index;
            }

            // anyway:
            state.apply_actions(state.row_actions[row_idx], state.col_actions[col_idx]);
            state.get_actions();
        }
    }

    void print(torch::Tensor t)
    {
        for (auto d : t.sizes())
        {
            std::cout << d << ' ';
        }
        std::cout << std::endl;
    }

    void learner()
    {
        // #ifdef ENABLE_TORCH

        DeviceBuffers temp_buffers{learner_buffer_size};

        while (!train)
        {
            sleep(1);
        }

        while (train)
        {
            torch::optim::SGD optimizer{net.parameters(), .001};
            torch::nn::MSELoss mse{};
            torch::nn::CrossEntropyLoss cel{};

            optimizer.zero_grad();
            try
            {
                learner_fetch();
                torch::Tensor jpi =
                    torch::from_blob(
                        learner_buffers.joined_policy_index_buffer,
                        {learner_buffer_size, 18}, {torch::kInt64})
                        .to(torch::kCUDA);
                // torch::Tensor f_i =
                //     torch::from_blob(
                //         learner_buffers.float_input_buffer,
                //         {learner_buffer_size, n_bytes_battle}).to(net.device).clone()
                //         .to(net.device);
                // torch::Tensor value_data =
                //     torch::from_blob(
                //         learner_buffers.value_data_buffer,
                //         {learner_buffer_size, 2}).to(net.device).clone()
                //         .to(net.device);
                // torch::Tensor value_target = value_data.index({"...", torch::indexing::Slice{0, 1, 1}});
                torch::cuda::synchronize();

                torch::Tensor jpi_ = jpi.clone().to(torch::kCUDA);
                torch::cuda::synchronize();

                // for (int i = 0; i < learner_buffer_size; ++i) {
                //     std::cout << jpi[i] << ' ';
                // }
                // std::cout << std::endl;
                // torch::Tensor joined_policy_target =
                //     torch::from_blob(
                //         learner_buffers.joined_policy_buffer,
                //         {learner_buffer_size, 18}).to(net.device).clone()
                //         .to(net.device);
                // torch::cuda::synchronize();
                // torch::Tensor sample__ =
                //     torch::from_blob(
                //         sample_buffers.joined_policy_index_buffer,
                //         {sample_buffer_size, 18}, torch::kInt64)
                //         .to(net.device);
                // torch::Tensor sample_buffer_index_tensor =
                //     torch::from_blob(
                //         sample_buffers.joined_policy_index_buffer,
                //         {sample_buffer_size, 18}, {torch::kInt64});
                std::cout << "entire learner policy index " << std::endl;
                pt(jpi_);

                torch::Tensor float_input = torch::rand({learner_buffer_size, 376}).to(torch::kCUDA);
                torch::Tensor value_target = torch::rand({learner_buffer_size, 1}).to(torch::kCUDA);

                torch::Tensor joined_policy_indices = torch::ones({learner_buffer_size, 18}, torch::kInt64).to(torch::kCUDA);

                torch::Tensor joined_policy_target = torch::rand({learner_buffer_size, 18}).to(torch::kCUDA);
                torch::Tensor row_policy_target = joined_policy_target.index({"...", torch::indexing::Slice{0, 9, 1}});
                torch::Tensor col_policy_target = joined_policy_target.index({"...", torch::indexing::Slice{9, 18, 1}});
                row_policy_target /= row_policy_target.norm(1, {1}, true);
                col_policy_target /= col_policy_target.norm(1, {1}, true);

                auto output = net.forward(float_input, jpi);

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
            }
            catch (const std::exception &e)
            {
                // Catch and handle any exceptions that might occur during the training loop
                std::cerr << "Exception caught: " << e.what() << std::endl;
            }

            float rate = learn_metric.update_and_get_rate(1);

            while (rate > 20)
            {
                sleep(1);
                rate = learn_metric.update_and_get_rate(0);
            }
            // std::cout << "learn rate: " << rate << std::endl;
        }
        // #endif
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
    // training_workspace.train = false;
    // training_workspace.generate_samples = false;

    training_workspace.start(1);

    return 0;
}
