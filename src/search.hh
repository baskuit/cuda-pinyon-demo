#include "pinyon.hh"

template <typename Types>
struct FlatSearch : Types
{

    static size_t hash_logs(const Types::Obs &obs)
    {

        const uint64_t *a = reinterpret_cast<const uint64_t *>(obs.get().data());
        size_t hash = 0;
        for (int i = 0; i < 8; ++i)
        {
            hash ^= a[i];
        }
        return hash;
    }

    class Search : public Types::BanditAlgorithm
    {
    public:
        using Types::BanditAlgorithm::BanditAlgorithm;
        Search() {}
        // template param?
        static const size_t max_iterations = (1 << 10);
        static const int max_iteration_depth = 10;

        std::array<typename Types::MatrixStats, max_iterations> matrix_stats{};

        // reset every run call
        size_t _iteration = 0;

        bool info[2 * max_iterations];

        std::unordered_map<uint64_t, int> transition{};

        // reset every iteration
        int depth = 0;
        int current_index = 0;
        typename Types::ModelOutput leaf_output;
        int rows, cols, row_idx, col_idx;

        std::array<typename Types::Outcome, max_iteration_depth> outcomes{}; // indices, mu

        std::array<int, max_iteration_depth> matrix_indices{}; // 0, 1, 4,

        void run_for_iterations(
            const size_t iterations,
            Types::PRNG &device,
            const Types::State &state,
            Types::Model &model)
        {
            transition = std::unordered_map<uint64_t, int>{};
            memset(info, false, 2 * max_iterations * sizeof(bool));
            for (_iteration = 0; _iteration < iterations; ++_iteration)
            {
                typename Types::State state_copy = state;
                state_copy.randomize_transition(device);
                run_iteration(device, state_copy, model);
            }
        }

        void run_iteration(
            Types::PRNG &device,
            Types::State &state,
            Types::Model &model)
        {
            // model = typename Types::Model{0};
            depth = 0;
            current_index = 0;

            // side by side: was_seen and is_expanded (state, e.g. expanded vectors)
            bool *info_ptr = &info[2 * current_index];

            while (*info_ptr && !state.is_terminal() && depth < max_iteration_depth)
            {
                typename Types::Outcome &outcome = outcomes[depth];
                typename Types::MatrixStats &stats = matrix_stats[current_index];

                // not really expanded
                if (!*(info_ptr + 1))
                {
                    rows = state.row_actions.size();
                    cols = state.col_actions.size();
                    this->expand_state_part(stats, rows, cols);
                    *(info_ptr + 1) = true;
                }

                this->select(device, stats, outcome);

                state.apply_actions(
                    state.row_actions[outcome.row_idx],
                    state.col_actions[outcome.col_idx]);
                state.get_actions();

                ++depth;
                uint64_t hash_ = hash(current_index, row_idx, col_idx, hash_logs(state.obs));

                if (transition[hash_] == 0)
                {
                    current_index = _iteration;
                    transition[hash_] = _iteration;
                    info_ptr = &info[2 * current_index];
                    break;
                }
                else
                {
                    current_index = transition[hash_];
                    info_ptr = &info[2 * current_index];
                }
                matrix_indices[depth] = current_index;
            }

            if (state.is_terminal())
            {
                leaf_output.value = state.payoff;
            }
            else
            {
                *info_ptr = true;
                model.inference(std::move(state), leaf_output);
            }
            this->expand_inference_part(matrix_stats[current_index], leaf_output);

            for (int d = 0; d < depth; ++d)
            {
                outcomes[d].value = leaf_output.value;
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
            int64_t lowPart = static_cast<int64_t>(hash);
            int64_t highPart = static_cast<int64_t>(hash >> 32);

            hashValue = hashValue * 31 + lowPart;
            hashValue = hashValue * 31 + highPart;

            return hashValue;
        }
    };
};
