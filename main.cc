#include <pinyon.hh>
#include <pkmn.h>

#include "./src/battle.hh"
#include "./src/common.hh"

// Pinyon type list for the entire program: Single threaded search on battles using Monte-Carlo eval.
using Types = TreeBandit<Exp3<MonteCarloModel<BattleTypes>>, DefaultNodes>;

void time_search()
{
    Types::PRNG device{0};
    Types::State state{0};
    Types::Model model{0};
    const Types::Search search{};
    Types::MatrixNode root{};

    const size_t iterations = 1 << 5;
    const size_t duration_ms = search.run_for_iterations(iterations, device, state, model, root);
    std::cout << iterations << " MCTS iterations completed in " << duration_ms << " ms." << std::endl;
}

void rollouts()
{
    Types::PRNG device{0};
    Types::Model model{0};
    const Types::Search search{};
    Types::MatrixNode root{};

    for (int i = 0; i < 100; ++i)
    {
        Types::State state{0};
        Types::ModelOutput o{};
        model.inference(std::move(state), o);
    }
}

size_t get_max_log()
{
    size_t max_log = 0;
    Types::PRNG device{};

    const size_t n_games = 100;
    const size_t n_tries_per_turn = 100;
    for (size_t i = 0; i < n_games; ++i)
    {
        Types::State state{i};
        Types::Action row_action, col_action;

        while (!state.is_terminal())
        {
            row_action = state.row_actions[device.random_int(state.row_actions.size())];
            col_action = state.col_actions[device.random_int(state.col_actions.size())];

            for (size_t j = 0; j < n_tries_per_turn; ++j)

            {
                Types::State state_copy{};
                state_copy.randomize_transition(device);
                state_copy.apply_actions(row_action, col_action);
                for (int len = log_size - 1; len > max_log; --len)
                {
                    if (state.log_options.buf[len] != 0)
                    {
                        if (len > max_log)
                        {
                            max_log = len;
                        }
                        break;
                    }
                }
            }
            state.apply_actions(row_action, col_action);
            state.get_actions();
        }
    }
    std::cout << max_log << std::endl;
    return max_log;
}

// RAII pinned buffer
struct ProducerBuffer
{
    ProducerBuffer(const size_t block_size, const size_t n_blocks)
        : block_size{block_size}, n_blocks{n_blocks}, batch_size{n_blocks * block_size}
    {
        alloc_buffers(&data.raw_input_buffer, &data.float_input_buffer, batch_size);
    }

    ProducerBuffer(const size_t batch_size)
        : block_size{batch_size}, n_blocks{1}, batch_size{batch_size}
    {
        alloc_buffers(&data.raw_input_buffer, &data.float_input_buffer, batch_size);
    }

    ~ProducerBuffer()
    {
        dealloc_buffers(&data.raw_input_buffer, &data.float_input_buffer);
    }

    ProducerBuffer(const ProducerBuffer &) = delete;

    const size_t block_size;
    const size_t n_blocks;
    const size_t batch_size;
    std::atomic<uint64_t> atomic_index{0};

    BufferData data{};

    size_t get_batch_index()
    {
        return atomic_index.fetch_add(1) % batch_size;
    }
};

struct ConsumerBuffer
{
    ConsumerBuffer(const size_t batch_size)
        : batch_size{batch_size}
    {
        alloc_buffers(&data.raw_input_buffer, &data.float_input_buffer, batch_size);
    }

    ~ConsumerBuffer()
    {
        dealloc_buffers(&data.raw_input_buffer, &data.float_input_buffer);
    }

    ConsumerBuffer(const ConsumerBuffer &) = delete;

    const size_t batch_size;
    std::atomic<uint64_t> atomic_index{0};

    BufferData data{};

    size_t get_batch_index()
    {
        return atomic_index.fetch_add(1) % batch_size;
    }
};

struct TrainingPool
{
    const size_t block_size;
    const size_t n_blocks;
    const size_t n_producer_blocks;
    const size_t n_prng_cuda_blocks;

    const size_t size;

    const size_t margin = 1; // number of blocks to avoid from up next
    // has own sync mechanisms
    ProducerBuffer producer_buffer{
        block_size, n_producer_blocks};

    const size_t learner_minibatch_size = 512;
    ProducerBuffer learner_buffer{
        learner_minibatch_size, 1};

    BufferData consumer_buffer_data{};

    std::atomic_int64_t atomic_block_index{0};

    std::vector<std::atomic_int64_t> block_atomics;
    Types::PRNG sample_device{};

    std::vector<uint64_t> raw_input_vector{};
    std::vector<float> float_input_vector{};
    std::vector<uint64_t> value_data_vector{};
    std::vector<float> joined_policy_vector{};
    std::vector<uint32_t> joined_policy_index_vector{};

    TrainingPool(const size_t block_size, const size_t n_blocks, const size_t n_producer_blocks, const size_t n_prng_cuda_blocks)
        : n_blocks{n_blocks}, n_producer_blocks{n_producer_blocks}, block_size{block_size}, size{n_blocks * block_size}, n_prng_cuda_blocks{n_prng_cuda_blocks}
    {
        raw_input_vector.reserve(size * 47);
        float_input_vector.reserve(size * 374);
        value_data_vector.reserve(size * 1);
        joined_policy_vector.reserve(size * 18);
        joined_policy_index_vector.reserve(size * 18);
        consumer_buffer_data = {
            raw_input_vector.data(),
            float_input_vector.data(),
            value_data_vector.data(),
            joined_policy_vector.data(),
            joined_policy_index_vector.data()};
    }

    TrainingPool(const TrainingPool &) = delete;

    void actor_store(const Types::State &state)
    {
        const size_t producer_batch_index = producer_buffer.get_batch_index();
        const size_t producer_block_index = producer_batch_index / block_size;
        const uint64_t *ptr = reinterpret_cast<const uint64_t *>(state.battle.bytes);
        copy(
            producer_buffer.data.raw_input_buffer + producer_batch_index * 47,
            ptr,
            47);
        convert(
            producer_buffer.data.float_input_buffer + producer_batch_index * 376,
            producer_buffer.data.raw_input_buffer + producer_batch_index * 47);

        if ((producer_batch_index + 1) % block_size == 0)
        {

            // store the 2nd most recent producer block
            const int src_block_index = (producer_block_index + n_producer_blocks - 1) % n_producer_blocks;
            const int tgt_block_index = atomic_block_index.fetch_add(1) % n_blocks;
            copy(
                consumer_buffer_data.raw_input_buffer + block_size * tgt_block_index * 376,
                producer_buffer.data.raw_input_buffer + block_size * src_block_index * 376,
                block_size);
            std::cout << "copy producer block: " << src_block_index << " to consumer block: " << tgt_block_index << std::endl;
        }
    }

    void learn_fetch(BufferData &learn_buffer_data)
    {
        const size_t n_samples_per_block = learner_minibatch_size / n_prng_cuda_blocks;
        const size_t start = (atomic_block_index.load() + n_blocks - n_prng_cuda_blocks) % n_blocks;

        sample(learn_buffer_data, consumer_buffer_data, block_size, start, n_prng_cuda_blocks, n_samples_per_block);
        // std::cout << "sample at " << start << std::endl;
        // foo (tgt, src, start, end, block_size, num_samples)
    };
};

void actor(
    TrainingPool *training_pool_ptr,
    const long long int max_counter)
{
    TrainingPool &training_pool = *training_pool_ptr;
    ProducerBuffer &buffer = training_pool.producer_buffer;

    Types::PRNG device{};
    Types::State state{device.get_seed()};
    Types::Model model{0};
    const Types::Search search{};
    const float full_prob = .25;

    int counter = 0;

    uint64_t index = buffer.atomic_index.fetch_add(1) % buffer.batch_size;
    while (counter < max_counter)
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
            ++counter;
            training_pool.actor_store(state);
        }
    }
}

void learner(
    TrainingPool *training_pool_ptr)
{
    while (true)
    {
        training_pool_ptr->learn_fetch(training_pool_ptr->learner_buffer.data);
    }
}

void buffer_thread_test()
{
    const int block_size = 1 << 10;
    const int n_consumer_blocks = 1 << 6;
    const int n_producer_blocks = 1 << 2;
    const int n_prng_cuda_blocks = 8;
    setup_rng(n_prng_cuda_blocks);

    TrainingPool training_pool{block_size, n_consumer_blocks, n_producer_blocks, n_prng_cuda_blocks};

    const size_t n_actor_threads = 6;
    std::thread threads[n_actor_threads];
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n_actor_threads; ++i)
    {
        threads[i] = std::thread(&actor, &training_pool, block_size * n_producer_blocks / n_actor_threads);
    }

    // std::thread learner_thread{&learner, &training_pool};

    for (int i = 0; i < n_actor_threads; ++i)
    {
        threads[i].join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << training_pool.producer_buffer.batch_size / (double)duration.count() * 1000 << " steps/sec" << std::endl;

    // learner_thread.join();
}

int main()
{
    // time_search();
    buffer_thread_test();
    return 0;
}
