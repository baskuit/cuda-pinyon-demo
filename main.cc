#include <pinyon.hh>
#include <pkmn.h>

#include "./src/battle.hh"
#include "./src/common.hh"

// Pinyon type list for the entire program: Single threaded search on battles using Monte-Carlo eval.
using Types = TreeBandit<Exp3<MonteCarloModel<BattleTypes>>, FlatNodes>;

void time_search()
{
    Types::PRNG device{0};
    Types::State state{0};
    Types::Model model{0};
    const Types::Search search{};
    Types::MatrixNode root{};

    const size_t iterations = 1 << 15;
    const size_t duration_ms = search.run_for_iterations(iterations, device, state, model, root);
    std::cout << iterations << " MCTS iterations completed in " << duration_ms << " ms." << std::endl;
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

struct Buffer
{

    Buffer(const size_t batch_size) : batch_size{batch_size}
    {
        alloc_buffers(&raw_input_buffer, &float_input_buffer, batch_size);
        std::cout << "raw input buffer size: " << (float(batch_size * 47 * 64) / 1000000) << std::endl;
        std::cout << "float input buffer size: " << (float(batch_size * 376 * 4) / 1000000) << std::endl;
        // std::cout << "raw input buffer size: " << << std::endl;
    }

    ~Buffer()
    {
        dealloc_buffers(&raw_input_buffer, &float_input_buffer);
    }

    const size_t batch_size;
    std::atomic<uint64_t> batch_idx{0};

    unsigned long long *raw_input_buffer;
    float *float_input_buffer;
    float *joined_policy_buffer;

    void store(Types::State &state, int index)
    {
        unsigned long long *ptr = reinterpret_cast<unsigned long long *>(state.battle.bytes);
        copy_battle(raw_input_buffer, ptr, index);
        convert(float_input_buffer, raw_input_buffer, index);
    }
};

void write_to_buffer(
    Buffer *buffer_ptr)
{
    Buffer &buffer = *buffer_ptr;

    Types::PRNG device{};
    Types::State state{};
    Types::Model model{0};
    const size_t iterations = 1 << 10;
    const Types::Search search{};

    uint64_t index = buffer.batch_idx.fetch_add(1);
    while (index < buffer.batch_size)
    {
        if (state.is_terminal())
        {
            state = Types::State{};
            state.randomize_transition(device);
        }
        state.get_actions();

        Types::MatrixNode root{};
        search.run_for_iterations(iterations, device, state, model, root);

        Types::VectorReal row_strategy, col_strategy;
        search.get_empirical_strategies(root.stats, row_strategy, col_strategy);
        const int row_idx = device.random_int(state.row_actions.size());
        const int col_idx = device.random_int(state.col_actions.size());

        state.apply_actions(state.row_actions[row_idx], state.col_actions[col_idx]);

        buffer.store(state, index);
        index = buffer.batch_idx.fetch_add(1);
    }
}

void buffer_thread_test()
{
    const int batch_size = 1 << 6;

    Buffer buffer{batch_size};

    const size_t n_threads = 1;
    std::thread threads[n_threads];
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n_threads; ++i)
    {
        threads[i] = std::thread(&write_to_buffer, &buffer);
    }

    for (int i = 0; i < n_threads; ++i)
    {
        threads[i].join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << batch_size / (double)duration.count() * 1000 << " steps/sec" << std::endl;
}
int main()
{
    buffer_thread_test();
    return 0;
}
