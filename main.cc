#include <pinyon.hh>
#include <pkmn.h>
// #include <torch/torch.h>

#include <queue>

#include "./src/battle.hh"
#include "./src/common.hh"
#include "./src/buffers.hh"
// #include "./src/net.hh"

// Pinyon type list for the entire program: Single threaded search on battles using Monte-Carlo eval.
using Types = TreeBandit<Exp3<MonteCarloModel<BattleTypes>>, DefaultNodes>;
// Need Types defined first
#include "./src/scripts.hh"

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

struct Net
{
};

template <typename Net>
struct Training
{
    Net net;

    bool train = false;
    bool generate_samples = true;

    const int sample_buffer_size;
    const int learner_buffer_size;

    PinnedBuffers sample_buffers{
        sample_buffer_size};
    std::atomic<uint64_t> sample_index{0};

    DeviceBuffers learner_buffers{
        learner_buffer_size};

    const float full_search_prob = .25;

    struct Metric
    {
        std::queue<int> donation_sizes;
        std::queue<int> donation_times;

        float update_and_get_rate()
        {
            return 0;
        }
    };

    Metric metric;

    template <typename... Args>
    Training(
        Args... args,
        const int sample_buffer_size,
        const int learner_buffer_size)
        : net{args...},
          sample_buffer_size{sample_buffer_size},
          learner_buffer_size{learner_buffer_size}
    {
        // net.to(torch::kCUDA);
    }

    Training(const Training &) = delete;

    void actor_store(HostBuffers &actor_buffers)
    {
        const int sample_index_last = sample_index.fetch_add(actor_buffers.size);
        const int sample_index_first = sample_index_last - actor_buffers.size;

        if (sample_index_first < sample_index_last)
        {
        }
        else
        {
        }
    }

    void learner_fetch(){};

    void actor()
    {
        // most samples possible per game
        const int max_samples = 500;
        // current index on actor buffer
        HostBuffers buffers{max_samples};

        Types::PRNG device{};
        Types::State state{device.get_seed()};
        Types::Model model{0};
        const Types::Search search{};
        uint32_t rows, cols;
        Types::VectorReal row_strategy, col_strategy;

        while (generate_samples)
        {
            if (state.is_terminal())
            {

                actor_store(buffers);
                buffers.size = 0;
                state = Types::State{device.get_seed()};
                state.randomize_transition(device);
            }
            Types::MatrixNode root{};
            const bool use_full_search = device.uniform() < full_search_prob;
            const size_t iterations = use_full_search ? 1 << 10 : 1 << 8;
            search.run_for_iterations(iterations, device, state, model, root);

            search.get_empirical_strategies(root.stats, row_strategy, col_strategy);

            rows = state.row_actions.size();
            cols = state.col_actions.size();
            const int row_idx = device.random_int(rows);
            const int col_idx = device.random_int(cols);

            if (use_full_search)
            {
                // battle
                memcpy(
                    buffers.raw_input_buffer + buffers.size * 47,
                    reinterpret_cast<const uint64_t *>(state.battle.bytes),
                    376);
                // value
                search.get_empirical_value(root.stats, *reinterpret_cast<Types::Value *>(buffers.value_data_buffer));
                // policy
                memcpy(
                    buffers.joined_policy_buffer + buffers.size * 18 * 4,
                    reinterpret_cast<float *>(row_strategy.data()),
                    rows * 4);
                memcpy(
                    buffers.joined_policy_buffer + buffers.size * 18 * 4 + 9 * 4,
                    reinterpret_cast<float *>(col_strategy.data()),
                    cols * 4);
                // joined actions
                memcpy(
                    buffers.joined_actions_buffer + buffers.size * 18,
                    reinterpret_cast<uint8_t *>(state.row_actions.data()),
                    rows * 4);
                memcpy(
                    buffers.joined_actions_buffer + buffers.size * 18 + 9,
                    reinterpret_cast<uint8_t *>(state.col_actions.data()),
                    cols * 4);
                ++buffers.size;
            }

            state.apply_actions(state.row_actions[row_idx], state.col_actions[col_idx]);
            state.get_actions();
        }
    }

    void learner()
    {

        // torch::optim::SGD optimizer{net->parameters(), .001};
        float x[8];
        x[0] = 1.0;
        while (train)
        {
            learner_fetch();
            // torch::Tensor input =
            //     torch::from_blob(
            //         learner_buffers.float_input_buffer,
            //         {int{learner_buffer_size}, int{n_bytes_battle}})
            //         .to(net->device);
            // auto output = net->forward(input);
            // torch::nn::MSELoss mse{};
            // optimizer.zero_grad();
            // torch::Tensor loss = mse(output, *target);
            // loss.backward();
            // optimizer.step();
            // cuda_memcpy_float32_device_to_host(x, training_pool_ptr->learner_buffers.float_input_buffer, 4 * 3);
            for (int i = 0; i < 8; ++i)
            {
                std::cout << reinterpret_cast<uint32_t *>(x)[i] << ' ';
            }
            std::cout << std::endl;
            sleep(1);
        }
    }

    void start(const int n_actor_threads)
    {
        std::thread actor_threads[n_actor_threads];
        std::thread learner_thread{&Training<Net>::learner, this};
        for (int i = 0; i < n_actor_threads; ++i)
        {
            actor_threads[i] = std::thread(&Training<Net>::actor, this);
        }
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
    const int learner_minibatch_size = 256;

    Training<Net> training_workspace{sample_buffer_size, learner_minibatch_size};

    training_workspace.start(4);

    return 0;
}
