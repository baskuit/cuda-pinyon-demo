#include <pinyon.hh>
#include <pkmn.h>
#include <torch/torch.h>

#include "./src/battle.hh"
#include "./src/common.hh"
#include "./src/net.hh"

// Pinyon type list for the entire program: Single threaded search on battles using Monte-Carlo eval.
using Types = TreeBandit<Exp3<MonteCarloModel<BattleTypes>>, DefaultNodes>;
// Need Types defined first
#include "./src/scripts.hh"

// RAII pinned buffer
struct PinnedBuffer : BufferData
{
    const size_t size;

    PinnedBuffer(const size_t size)
        : size{size}
    {
        alloc_buffers(*this, size);
    }

    ~PinnedBuffer()
    {
        dealloc_buffers(*this);
    }

    PinnedBuffer(const PinnedBuffer &) = delete;
    PinnedBuffer &operator=(const PinnedBuffer &&) = delete;
};

struct CPUBuffer : BufferData
{
    const size_t size;
    std::vector<uint64_t> raw_input_vector{};
    std::vector<float> float_input_vector{};
    std::vector<uint64_t> value_data_vector{};
    std::vector<float> joined_policy_vector{};
    std::vector<uint32_t> joined_policy_index_vector{};

    CPUBuffer(const size_t size)
        : BufferData{
              raw_input_vector.data(),
              float_input_vector.data(),
              value_data_vector.data(),
              joined_policy_vector.data(),
              joined_policy_index_vector.data()},
          size{size}
    {
        raw_input_vector.reserve(size * 47);
        float_input_vector.reserve(size * 374);
        value_data_vector.reserve(size * 1);
        joined_policy_vector.reserve(size * 18);
        joined_policy_index_vector.reserve(size * 18);
    }

    CPUBuffer(const CPUBuffer &) = delete;
    CPUBuffer &operator=(const CPUBuffer &&) = delete;
};

/*
Initializses buffers, contains synch mechanism for actor and learner threads
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

    PinnedBuffer producer_buffer{
        producer_buffer_size};
    std::atomic<uint64_t> producer_index{0};

    PinnedBuffer learner_buffer{
        learner_minibatch_size};

    CPUBuffer consumer_buffer{
        block_size * n_consumer_blocks};
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

    void actor_store(const Types::State &state) // TODO add policy data, etc
    {
        const size_t producer_buffer_index = producer_index.fetch_add(1) % producer_buffer_size;
        const size_t producer_block_index = producer_buffer_index / block_size;
        const uint64_t *ptr = reinterpret_cast<const uint64_t *>(state.battle.bytes);
        copy(
            producer_buffer.raw_input_buffer + producer_buffer_index * 47,
            ptr,
            47);
        convert(
            producer_buffer.float_input_buffer + producer_buffer_index * 376,
            producer_buffer.raw_input_buffer + producer_buffer_index * 47);

        if ((producer_buffer_index + 1) % block_size == 0)
        {
            // store the 2nd most recent producer block
            const size_t src_block_index = (producer_block_index + n_producer_blocks - 1) % n_producer_blocks;
            const size_t tgt_block_index = consumer_block_index.fetch_add(1) % n_consumer_blocks;
            copy(
                consumer_buffer.raw_input_buffer + block_size * tgt_block_index * 376,
                producer_buffer.raw_input_buffer + block_size * src_block_index * 376,
                block_size);
            std::cout << "copy producer block: " << src_block_index << " to consumer block: " << tgt_block_index << std::endl;
        }
    }

    void learn_fetch(BufferData &learn_buffer_data)
    {
        const size_t n_samples_per_block = learner_minibatch_size / n_prng_cuda_blocks;
        const size_t start = (consumer_block_index.load() + n_consumer_blocks - n_prng_cuda_blocks) % n_consumer_blocks;
        sample(learn_buffer_data, consumer_buffer, block_size, start, n_prng_cuda_blocks, n_samples_per_block);
    };
};

void actor(
    TrainingPool *training_pool_ptr)
{
    TrainingPool &training_pool = *training_pool_ptr;
    PinnedBuffer &buffer = training_pool.producer_buffer;

    Types::PRNG device{};
    Types::State state{device.get_seed()};
    Types::Model model{0};
    const Types::Search search{};
    const float full_prob = .25;

    while (training_pool.producer_index.load() < training_pool.block_size * training_pool.n_producer_blocks)
    {
        if (state.is_terminal())
        {
            state = Types::State{device.get_seed()};
            state.randomize_transition(device);
        }
        Types::MatrixNode root{};
        const bool use_full = device.uniform() < full_prob;
        const size_t iterations = use_full ? 1 << 10 : 1 << 8;
        search.run_for_iterations(iterations, device, state, model, root);

        Types::VectorReal row_strategy, col_strategy;
        search.get_empirical_strategies(root.stats, row_strategy, col_strategy);
        const int row_idx = device.random_int(state.row_actions.size());
        const int col_idx = device.random_int(state.col_actions.size());

        state.apply_actions(state.row_actions[row_idx], state.col_actions[col_idx]);
        state.get_actions();

        if (use_full)
        {
            training_pool.actor_store(state);
        }
    }
}

void learner(
    TrainingPool *training_pool_ptr,
    Net *net,
    torch::Tensor *input,
    torch::Tensor *target)
{
    torch::optim::SGD optimizer{net->parameters(), .001};
    while (true)
    {
        training_pool_ptr->learn_fetch(training_pool_ptr->learner_buffer);
        // torch::Tensor input = torch::from_blob(
        //     training_pool_ptr->float_input_vector.data(),
        //     {training_pool_ptr->learner_minibatch_size, 376});
        auto output = net->forward(*input);
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
    const int block_size = 1 << 10;
    const int n_consumer_blocks = 1 << 6;
    const int n_producer_blocks = 1 << 2;
    const int n_prng_cuda_blocks = 8;
    const int learner_minibatch_size = 256;
    // setup_rng(n_prng_cuda_blocks);
    TrainingPool training_pool{block_size, n_consumer_blocks, n_producer_blocks, n_prng_cuda_blocks, learner_minibatch_size};

    std::cout << "is cuda available" << torch::cuda::is_available() << std::endl;
    auto net = Net();
    net.to(torch::kCUDA);
    torch::Tensor input = torch::rand({learner_minibatch_size, 376}).to(net.device);
    torch::Tensor target = torch::rand({learner_minibatch_size, 1}).to(net.device);

    // setup_rng(n_prng_cuda_blocks);

    const size_t n_actor_threads = 6;
    std::thread threads[n_actor_threads];
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n_actor_threads; ++i)
    {
        threads[i] = std::thread(&actor, &training_pool);
    }

    std::thread learner_thread{&learner, &training_pool, &net, &input, &target};

    for (int i = 0; i < n_actor_threads; ++i)
    {
        threads[i].join();
    }

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
