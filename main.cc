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

    Buffer(const size_t batch_size, const size_t input_size) : batch_size{batch_size}, input_size{input_size}
    {
        alloc_buffers(&raw_input_buffer, &float_input_buffer, batch_size);
    }

    ~Buffer()
    {
        dealloc_buffers(&raw_input_buffer, &float_input_buffer);
    }

    const size_t batch_size;
    const size_t input_size;
    unsigned long long *raw_input_buffer;
    float *float_input_buffer;

    void store(Types::State &state, int index)
    {
        unsigned long long *ptr = reinterpret_cast<unsigned long long *>(state.battle.bytes);
        copy_battle(raw_input_buffer + (index * 48), ptr, index);
        convert(raw_input_buffer + (index * 48), float_input_buffer + (index * 384));
    }
};

void write_to_buffer()
{
    const int batch_size = 1 << 10;

    Buffer buffer{batch_size, 386};

    Types::PRNG device{};
    Types::State state{device.get_seed()};
    Types::Model model{0};
    const size_t iterations = 1 << 6;

    const Types::Search search{};
    auto start = std::chrono::high_resolution_clock::now();


    for (int index = 0; index < batch_size; ++index)
    {

        if (state.is_terminal())
        {
            state = Types::State{device.get_seed()};
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
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << batch_size / (double) duration.count() << " steps/ms" << std::endl;

    // for (int i = 0; i < 384; i += 1)
    // {
    //     std::cout << buffer.float_input_buffer[i] << ' ';
    // }
    // std::cout << std::endl;
}
int main()
{
    write_to_buffer();
    return 0;
}
