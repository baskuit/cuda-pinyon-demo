#include <pinyon.hh>
#include <pkmn.h>
#include <torch/torch.h>

#include "./src/battle.hh"
#include "./src/common.hh"
#include "./src/buffers.hh"
#include "./src/net.hh"

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
The program exploits the pinning with a CUDA kernel that converts the `uint64_t[47]` battle data into `float[n_bytes_input]` data

The producer buffer and the learner buffer are split into uniform 'blocks' (of 'block_size' many samples) for transfer.
(Each sample has byte size which is the sum of the value, policy, etc byte sizes.
For example the raw_input part of a single sample is n_bytes_input bytes, and its float version is n_bytes_input * 4 bytes.)
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

struct TrainingPool
{
    const size_t block_size;
    const size_t n_consumer_blocks;
    const size_t n_producer_blocks;
    const size_t n_prng_cuda_blocks;
    const size_t learner_minibatch_size;

    const size_t producer_buffer_size{block_size * n_producer_blocks};
    const size_t consumer_buffer_size{block_size * n_consumer_blocks};

    PinnedBuffers producer_buffers{
        producer_buffer_size};
    std::atomic<uint64_t> producer_index{0};

    DeviceBuffers learner_buffers{
        learner_minibatch_size};

    PinnedBuffers consumer_buffers{
        consumer_buffer_size};
    std::atomic_int64_t consumer_block_index{0};

    TrainingPool(
        const size_t block_size,
        const size_t n_consumer_blocks,
        const size_t n_producer_blocks,
        const size_t n_prng_cuda_blocks,
        const size_t learner_minibatch_size)
        : block_size{block_size},
          n_consumer_blocks{n_consumer_blocks},
          n_producer_blocks{n_producer_blocks},
          n_prng_cuda_blocks{n_prng_cuda_blocks},
          learner_minibatch_size{learner_minibatch_size}
    {
    }

    TrainingPool(const TrainingPool &) = delete;

    void actor_store(ActorBuffers &actor_buffers, const size_t buffer_size)
    {
        const size_t producer_buffer_index_last = producer_index.fetch_add(buffer_size) % producer_buffer_size;
        const size_t producer_buffer_index_first = (producer_buffer_index_last + producer_buffer_size - buffer_size + 1) % producer_buffer_size;
        const size_t producer_block_index_last = producer_buffer_index_last / block_size;
        const size_t producer_block_index_first = producer_buffer_index_first / block_size;

        copy(
            producer_buffers.raw_input_buffer + producer_buffer_index_first * 47,
            actor_buffers.raw_input_buffer,
            47 * buffer_size);
        convert(
            producer_buffers.float_input_buffer + producer_buffer_index_first * n_bytes_input,
            producer_buffers.raw_input_buffer + producer_buffer_index_first * 47);
    }

    void actor_store(const Types::State &state) // TODO add policy data, etc
    {
        const size_t producer_buffer_index = producer_index.fetch_add(1) % producer_buffer_size;
        const size_t producer_block_index = producer_buffer_index / block_size;
        const uint64_t *ptr = reinterpret_cast<const uint64_t *>(state.battle.bytes);
        copy(
            producer_buffers.raw_input_buffer + producer_buffer_index * 47,
            ptr,
            47);
        convert(
            producer_buffers.float_input_buffer + producer_buffer_index * n_bytes_input,
            producer_buffers.raw_input_buffer + producer_buffer_index * 47);

        if ((producer_buffer_index + 1) % block_size == 0)
        {
            // store the 2nd most recent producer block
            const size_t src_block_index = (producer_block_index + n_producer_blocks - 1) % n_producer_blocks;
            const size_t tgt_block_index = consumer_block_index.fetch_add(1) % n_consumer_blocks;
            copy(
                consumer_buffers.raw_input_buffer + block_size * tgt_block_index * n_bytes_input,
                producer_buffers.raw_input_buffer + block_size * src_block_index * n_bytes_input,
                block_size * n_bytes_input);
            std::cout << "copy producer block: " << src_block_index << " to consumer block: " << tgt_block_index << std::endl;
        }
    }

    void learner_fetch()
    {
        const size_t n_samples_per_block = learner_minibatch_size / n_prng_cuda_blocks;
        assert(n_prng_cuda_blocks * n_samples_per_block == learner_minibatch_size);
        const size_t start = (consumer_block_index.load() + n_consumer_blocks - n_prng_cuda_blocks) % n_consumer_blocks;
        std::cout << "learn_fetch: n_samples_per_block " << n_samples_per_block << ", start " << start << std::endl;
        sample(learner_buffers, consumer_buffers,
               block_size, n_consumer_blocks, start, n_prng_cuda_blocks, n_samples_per_block);
    };
};

void actor(
    TrainingPool *training_pool_ptr)
{
    Types::PRNG device{};
    Types::State state{device.get_seed()};
    Types::Model model{0};
    const Types::Search search{};
    const float full_prob = .25;

    HostBuffers game_buffers{1000};
    game_buffers.size = 0;
    uint32_t rows, cols;
    Types::VectorReal row_strategy, col_strategy;

    while (training_pool_ptr->producer_index.load() < training_pool_ptr->block_size * training_pool_ptr->n_producer_blocks)
    {
        if (state.is_terminal())
        {

            training_pool_ptr->actor_store(game_buffers, game_buffers.size);
            game_buffers.size = 0;

            state = Types::State{device.get_seed()};
            state.randomize_transition(device);
        }
        Types::MatrixNode root{};
        const bool use_full_iterations = device.uniform() < full_prob;
        const size_t iterations = use_full_iterations ? 1 << 10 : 1 << 8;
        search.run_for_iterations(iterations, device, state, model, root);

        search.get_empirical_strategies(root.stats, row_strategy, col_strategy);

        rows = state.row_actions.size();
        cols = state.col_actions.size();
        const int row_idx = device.random_int(rows);
        const int col_idx = device.random_int(cols);

        if (use_full_iterations)
        {
            // battle
            memcpy(
                game_buffers.raw_input_buffer + game_buffers.size * 47,
                reinterpret_cast<const uint64_t *>(state.battle.bytes),
                47);
            // value
            search.get_empirical_value(root.stats, *reinterpret_cast<Types::Value *>(game_buffers.value_data_buffer));
            // policy
            memcpy(
                game_buffers.joined_policy_buffer + game_buffers.size * 18 * 4,
                reinterpret_cast<float *>(row_strategy.data()),
                rows * 4);
            memcpy(
                game_buffers.joined_policy_buffer + game_buffers.size * 18 * 4 + 9 * 4,
                reinterpret_cast<float *>(col_strategy.data()),
                cols * 4);
            // joined actions
            memcpy(
                game_buffers.joined_actions_buffer + game_buffers.size * 18,
                reinterpret_cast<uint8_t *>(state.row_actions.data()),
                rows * 4);
            memcpy(
                game_buffers.joined_actions_buffer + game_buffers.size * 18 + 9,
                reinterpret_cast<uint8_t *>(state.col_actions.data()),
                cols * 4);
            // joined n actions
            game_buffers.joined_n_actions_buffer[game_buffers.size * 2] = rows;
            game_buffers.joined_n_actions_buffer[game_buffers.size * 2 + 1] = cols;

            ++game_buffers.size;
        }

        state.apply_actions(state.row_actions[row_idx], state.col_actions[col_idx]);
        state.get_actions();
    }
}

void learner(
    TrainingPool *training_pool_ptr,
    Net *net,
    torch::Tensor *target)
{
    torch::optim::SGD optimizer{net->parameters(), .001};
    while (true)
    {
        training_pool_ptr->learner_fetch();
        torch::Tensor input =
            torch::from_blob(
                training_pool_ptr->learner_buffers.float_input_buffer,
                {int{training_pool_ptr->learner_minibatch_size}, n_bytes_input})
                .to(net->device);
        auto output = net->forward(input);
        torch::nn::MSELoss mse{};
        optimizer.zero_grad();
        torch::Tensor loss = mse(output, *target);
        loss.backward();
        optimizer.step();
        std::cout << '!' << std::endl;
    }
}

void buffer_thread_test()
{
    const int block_size = 1 << 8;
    const int n_consumer_blocks = 1 << 8;
    const int n_producer_blocks = 1 << 4;
    const int n_prng_cuda_blocks = 1 << 7;
    const int learner_minibatch_size = 256;
    setup_rng(n_prng_cuda_blocks);
    TrainingPool training_pool{block_size, n_consumer_blocks, n_producer_blocks, n_prng_cuda_blocks, learner_minibatch_size};

    std::cout << "is cuda available" << torch::cuda::is_available() << std::endl;
    auto net = Net();
    net.to(torch::kCUDA);
    torch::Tensor target = torch::rand({learner_minibatch_size, 1}).to(net.device);

    // setup_rng(n_prng_cuda_blocks);

    const size_t n_actor_threads = 6;
    std::thread threads[n_actor_threads];
    auto start = std::chrono::high_resolution_clock::now();

    // for (int i = 0; i < n_actor_threads; ++i)
    // {
    //     threads[i] = std::thread(&actor, &training_pool);
    // }

    std::thread learner_thread{&learner, &training_pool, &net, &target};

    // for (int i = 0; i < n_actor_threads; ++i)
    // {
    //     threads[i].join();
    // }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << training_pool.producer_buffer_size / (double)duration.count() * 1000 << " steps/sec" << std::endl;

    learner_thread.join();
}

int main()
{
    buffer_thread_test();
    return 0;
}
