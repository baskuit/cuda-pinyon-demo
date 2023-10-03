#include "pinyon.hh"

template <typename Types>
struct Search
{
    // template param?
    const size_t max_iterations = (1 << 10);
    const int max_iteration_depth = 10;

    std::array<typename Types::MatrixStats, max_iterations> matrix_stats;

    // reset every run call
    size_t _iteration = 0;

    std::array<bool, 2 * max_iterations> info;

    std::unordered_map<uint64_t, int> transition{};

    // reset every iteration
    int depth = 0;
    int current_index = 0;
    typename Types::Value leaf_value;
    typename Types::ModelOutput leaf_output;
    int rows, cols, row_idx, col_idx;

    std::array<typename Types::Outcome, max_iteration_depth> outcomes; // indices, mu

    std::array<int, max_iteration_depth> matrix_indices; // 0, 1, 4,
    std::array<int, max_iteration_depth> chance_indices; // 0, 1, 6 etc

    void run_for_iterations(
        const size_t iterations,
        Types::PRNG &device,
        const Types::State &state,
        Types::Model &model)
    {
        transition = std::unordered_map<uint64_t, int>{};
        memset(info.data(), false, 2 * max_iterations * sizeof(bool));
        for (_iteration = 0; _iteration < iterations; ++_iteration)
        {
            typename Types::State state_copy = state;
            run_iteration(device, state_copy, model);
        }
    }

    void run_iteration(
        Types::PRNG &device,
        const Types::State &state,
        Types::Model &model)
    {
        depth = 0;
        current_index = 0;

        bool *info_ptr = info + 2 * current_index;

        while (*info_ptr && !state.is_terminal() && depth < max_iteration_depth)
        {
            typename Types::Outcome &outcome = outcomes[depth];
            typename Types::MatrixStats &stats = matrix_stats[current_index];

            if (!*(info_ptr + 1))
            {
                model.inference(state, leaf_output);
                rows = state.row_actions.size();
                cols = state.col_actions.size();
                this->expand(stats, rows, cols, leaf_output);
                *(info_ptr + 1) = true;
            }

            this->select(device, stats, outcome);

            state.apply_actions(
                state.row_actions[outcome.row_idx],
                state.col_actions[outcome.col_idx]);
            state.get_actions();

            uint64_t hash_ = hash(current_index, row_idx, col_idx, static_cast<uint64_t>(state.obs));
            current_index = transition.at(hash_);
            if (current_index == 0)
            {
                current_index = _iteration;
                transition.at(hash_) = _iteration;
                break;
            }
            ++depth;
        }

        if (state.is_terminal())
        {
            leaf_value = state.payoff;
        }
        else
        {
            *info_ptr = true;
            model.inference(std::move(state), leaf_output);
        }

        for (int d = 0; d < depth - 1; ++d)
        {
            this->update_matrix_stats(matrix_stats[matrix_indices[d]], outcomes[d]);
        }
    }

    uint64_t hash(int index, int row_idx, int col_idx, uint64_t hash)
    {
        size_t hashValue = 17; // Choose a prime number as the initial value

        // Hash the integers
        hashValue = hashValue * 31 + index;
        hashValue = hashValue * 31 + row_idx;
        hashValue = hashValue * 31 + col_idx;

        // Hash the uint64_t (hash)
        uint32_t lowPart = static_cast<uint32_t>(hash);
        uint32_t highPart = static_cast<uint32_t>(hash >> 32);

        hashValue = hashValue * 31 + lowPart;
        hashValue = hashValue * 31 + highPart;

        return hashValue;
    }
};
