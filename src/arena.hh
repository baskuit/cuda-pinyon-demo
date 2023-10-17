#include <pinyon.hh>

#include "./battle.hh"
#include "./nn.hh"
#include "./cpu-model.hh"

#include <utility>

struct NNArena : SimpleTypes
{
    using N = TreeBanditSearchModel<TreeBandit<Exp3<CPUModel<FCResNet>>>>;
    using M = TreeBanditSearchModel<TreeBandit<Exp3<MonteCarloModel<BattleTypes>>>>;

    class State : public PerfectInfoState<SimpleTypes>
    {
    public:
        BattleTypes::State (*state_generator)(SimpleTypes::Seed){nullptr};
        const std::vector<size_t> mcm_iter;
        const std::vector<N::Model> models{};
        const size_t vs_rounds;
        SimpleTypes::Seed state_seed{};

        State(
            BattleTypes::State (*state_generator)(SimpleTypes::Seed),
            const std::vector<size_t> &mcm_iter,
            const std::vector<N::Model> &models,
            size_t vs_rounds = 1)
            : state_generator{state_generator}, mcm_iter{mcm_iter}, models{models}, vs_rounds{vs_rounds}
        {
            this->init_range_actions(mcm_iter.size() + models.size());
        }

        void get_actions() const {}

        void get_actions(
            SimpleTypes::VectorAction &row_actions,
            SimpleTypes::VectorAction &col_actions) const
        {
            row_actions = this->row_actions;
            col_actions = this->col_actions;
        }

        void randomize_transition(SimpleTypes::PRNG &device)
        {
            state_seed = device.uniform_64();
        }

        void apply_actions(
            SimpleTypes::Action row_action,
            SimpleTypes::Action col_action)
        {

            BattleTypes::State state{state_generator(state_seed)};
            BattleTypes::PRNG device{state_seed};
            BattleTypes::Value p, q;

            std::cout << "apply_actions: " << row_action.get() << ' ' << col_action.get() << std::endl;

            bool row_easier = (row_action.get() == 0);

            if (row_action.get() < mcm_iter.size())
            {
                if (col_action.get() < mcm_iter.size())
                {
                    M::Model row_model{
                        mcm_iter[row_action.get()], {}, {device.uniform_64()}, {}};
                    M::Model col_model{
                        mcm_iter[col_action.get()], {}, {device.uniform_64()}, {}};
                    p = play<M, M>(
                        device, state,
                        row_model,
                        col_model,
                        row_easier);
                    q = play<M, M>(
                        device, state,
                        col_model,
                        row_model,
                        !row_easier);
                }
                else
                {
                    M::Model row_model{
                        mcm_iter[row_action.get()], {}, {device.uniform_64()}, {}};
                    N::Model col_model = models[col_action.get() - mcm_iter.size()];
                    p = play<M, N>(
                        device, state,
                        row_model,
                        col_model,
                        row_easier);
                    q = play<N, M>(
                        device, state,
                        col_model,
                        row_model,
                        !row_easier);
                }
            }
            else
            {
                if (col_action.get() < mcm_iter.size())
                {

                    N::Model row_model = models[row_action.get() - mcm_iter.size()];
                    M::Model col_model{mcm_iter[col_action.get()], {}, {device.uniform_64()}, {}};
                    p = play<N, M>(
                        device, state,
                        row_model,
                        col_model,
                        row_easier);
                    q = play<M, N>(
                        device, state,
                        col_model,
                        row_model,
                        !row_easier);
                }
                else
                {
                    N::Model row_model = models[row_action.get() - mcm_iter.size()];
                    N::Model col_model = models[col_action.get() - mcm_iter.size()];
                    p = play<N, N>(
                        device, state,
                        row_model,
                        col_model,
                        row_easier);
                    q = play<N, N>(
                        device, state,
                        col_model,
                        row_model,
                        !row_easier);
                }
            }
            double x = (p.get_row_value() + q.get_col_value()).get() / 2;
            double y = (q.get_row_value() + p.get_col_value()).get() / 2;
            this->payoff = SimpleTypes::Value{x, y};
            this->terminal = true;
            this->obs = SimpleTypes::Obs{static_cast<int>(device.get_seed())};
        }

        template <typename RowTypes, typename ColTypes>
        BattleTypes::Value play(
            BattleTypes::PRNG &device,
            const BattleTypes::State &state_,
            RowTypes::Model row_model,
            ColTypes::Model col_model,
            bool this_row_easier) const
        {
            int turn = 0;

            BattleTypes::State state{state_};
            state.randomize_transition(device);

            while (!state.is_terminal())
            {

                // std::cout << std::endl;
                // std::cout << "TURN: " << turn << std::endl;
                // state.print();
                // std::cout << std::endl;

                int row_idx = 0, col_idx = 0;
                if (state.row_actions.size() > 1)
                {
                    typename RowTypes::ModelOutput output{};
                    BattleTypes::State state_{state};
                    if (this_row_easier) {
                        state_.clamp = true;
                    }
                    row_model.inference(std::move(state_), output);
                    row_idx = device.sample_pdf(output.row_policy);
                    // std::cout << "row player data:" << std::endl;
                    // math::print(output.row_policy);
                    // math::print(output.col_policy);
                    // std::cout << output.value << std::endl;
                }
                if (state.col_actions.size() > 1)
                {
                    typename ColTypes::ModelOutput output{};
                    BattleTypes::State state_{state};
                    if (!this_row_easier) {
                        state_.clamp = true;
                    }
                    col_model.inference(std::move(state_), output);
                    col_idx = device.sample_pdf(output.col_policy);
                    // std::cout << "cow player data:" << std::endl;
                    // math::print(output.row_policy);
                    // math::print(output.col_policy);
                    // std::cout << output.value << std::endl;
                };

                const int r_idx = get_index_from_action(state.battle.bytes, state.row_actions[row_idx].get(), 0);
                const int c_idx = get_index_from_action(state.battle.bytes, state.col_actions[col_idx].get(), 184);
                auto s = id_strings[r_idx];
                auto t = id_strings[c_idx];

                // std::cout << s << ", " << t << std::endl;

                state.apply_actions(
                    state.row_actions[row_idx],
                    state.col_actions[col_idx]);
                state.get_actions();
                ++turn;
            }
            // std::cout << "State Payoff" << std::endl;
            // std::cout << state.payoff << std::endl;
            return state.payoff;
        }
    };
};

using ArenaS = TreeBanditThreaded<Exp3Single<EmptyModel<NNArena>>>;

void foo(
    const NNArena::State &arena,
    ArenaS::MatrixNode &root,
    const size_t threads = 4,
    const size_t iter = 64)
{
    if (!root.is_expanded())
    {
        const size_t rows = arena.row_actions.size();
        root.expand(rows, rows);
        auto &stats = root.stats;
        stats.visits.resize(rows, 0);
        stats.gains.resize(rows, 0);
        stats.joint_visits.fill(rows, rows);
        stats.cum_values.fill(rows, rows);
    }
    ArenaS::PRNG device{};
    ArenaS::Search search{{.01}, threads};
    ArenaS::Model model{};
    search.run_for_iterations(iter, device, arena, model, root);
}
