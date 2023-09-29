#pragma once

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
