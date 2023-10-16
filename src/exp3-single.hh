#include <libpinyon/math.hh>
#include <algorithm/algorithm.hh>
#include <tree/tree.hh>

/*

Modification of vanilla Exp3 for the case of one-shot symmetric games, e.g. Arena

There is only one set of gains, visits stats, and the select function is guaranteed not to produce mirror matches.

Arena is used with multithreaded bandit, so off-policy and singlethreaded functions were removed

*/

template <CONCEPT(IsValueModelTypes, Types)>
struct Exp3Single : Types
{

    using Real = typename Types::Real;

    struct MatrixStats
    {
        typename Types::VectorReal gains;
        typename Types::VectorInt visits;
        typename Types::MatrixReal joint_visits;
        typename Types::template Matrix<PairReal<Real>> cum_values;

        int n = 0;
        PairReal<Real> value_total{0, 0};
    };
    struct ChanceStats
    {
    };
    struct Outcome
    {
        int row_idx, col_idx;
        typename Types::Value value;
        Real row_mu, col_mu;
    };

    class BanditAlgorithm
    {
    public:
        const Real gamma{.01};
        const Real one_minus_gamma{gamma * -1 + 1};

        constexpr BanditAlgorithm() {}

        constexpr BanditAlgorithm(Real gamma) : gamma(gamma), one_minus_gamma{gamma * -1 + 1} {}

        friend std::ostream &operator<<(std::ostream &os, const BanditAlgorithm &search)
        {
            os << "Exp3Single; gamma: " << search.gamma;
            return os;
        }

        void get_empirical_strategies(
            const MatrixStats &stats,
            Types::VectorReal &row_strategy,
            Types::VectorReal &col_strategy) const
        {
            row_strategy.resize(stats.visits.size());
            math::power_norm(stats.visits, row_strategy.size(), 1, row_strategy);
            col_strategy = row_strategy;
        }

        void get_empirical_value(
            const MatrixStats &stats,
            Types::Value &value) const
        {
            const Real den = typename Types::Q{1, (stats.n + (stats.n == 0))};
            if constexpr (Types::Value::IS_CONSTANT_SUM)
            {
                value = typename Types::Value{typename Types::Real{stats.value_total.get_row_value() * den}};
            }
            else
            {
                value = typename Types::Value{stats.value_total * den};
            }
        }

        void expand(
            MatrixStats &stats,
            const size_t &rows,
            const size_t &cols,
            const Types::ModelOutput &output) const
        {
            stats.visits.resize(rows, 0);
            stats.gains.resize(rows, 0);
            stats.joint_visits.fill(rows, rows);
            stats.cum_values.fill(rows, rows);
        }

        // multithreaded

        void select(
            Types::PRNG &device,
            const MatrixStats &stats,
            Outcome &outcome) const
        {
            typename Types::VectorReal forecast(stats.gains);
            const size_t rows = stats.gains.size();
            assert(rows > 1);
            const auto &one_minus_gamma = this->one_minus_gamma;

            const Real eta{gamma / static_cast<Real>(rows)};
            softmax(forecast, stats.gains, rows, eta);
            std::transform(
                forecast.begin(), forecast.begin() + rows, forecast.begin(),
                [eta, one_minus_gamma](Real value)
                { return one_minus_gamma * value + eta; });

            const int row_idx = device.sample_pdf(forecast, rows);
            int col_idx = row_idx;
            while (col_idx == row_idx)
            {
                col_idx = device.sample_pdf(forecast, rows);
            }
            outcome.row_idx = row_idx;
            outcome.col_idx = col_idx;
            outcome.row_mu = forecast[row_idx];
            outcome.col_mu = forecast[col_idx] / (typename Types::Real{typename Types::Q{1}} - outcome.row_mu);
        }

        void update_matrix_stats(
            MatrixStats &stats,
            const Outcome &outcome,
            Types::Mutex &mutex) const
        {
            mutex.lock();
            stats.value_total += outcome.value;
            stats.n += 1;
            stats.visits[outcome.row_idx] += 1;
            stats.visits[outcome.col_idx] += 1;
            if ((stats.gains[outcome.row_idx] += outcome.value.get_row_value() / outcome.row_mu) >= 0)
            {
                const auto max = stats.gains[outcome.row_idx];
                for (auto &v : stats.gains)
                {
                    v -= max;
                }
            }
            if ((stats.gains[outcome.col_idx] += outcome.value.get_col_value() / outcome.col_mu) >= 0)
            {
                const auto max = stats.gains[outcome.col_idx];
                for (auto &v : stats.gains)
                {
                    v -= max;
                }
            }

            const int a = std::min(outcome.row_idx, outcome.col_idx);
            const int b = std::max(outcome.row_idx, outcome.col_idx);
            stats.joint_visits.get(a, b) += 1;
            if (outcome.row_idx < outcome.col_idx)
            {
                stats.cum_values.get(a, b) += outcome.value;
            }
            else
            {
                stats.cum_values.get(a, b) += typename Types::Value{outcome.value.get_col_value(), outcome.value.get_row_value()};
            }
            mutex.unlock();
        }

        void update_chance_stats(
            ChanceStats &stats,
            const Outcome &outcome) const
        {
        }

    private:
        inline void softmax(
            Types::VectorReal &forecast,
            const Types::VectorReal &gains,
            const size_t k,
            Real eta) const
        {
            Real sum = 0;
            for (size_t i = 0; i < k; ++i)
            {
                const Real y{std::exp(static_cast<typename Types::Float>(gains[i] * eta))};
                forecast[i] = y;
                sum += y;
            }
            for (size_t i = 0; i < k; ++i)
            {
                forecast[i] /= sum;
            }
        };
    };
};
