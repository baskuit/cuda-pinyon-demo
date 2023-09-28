#include <pinyon.hh>
#include <pkmn.h>

#include "./src/battle.hh"
#include "./src/common.hh"

// Pinyon type list for the entire program: Single threaded search on battles using Monte-Carlo eval.
using Types = TreeBandit<Exp3<MonteCarloModel<MoldState<>>>>;

int main () {

    Types::PRNG device{0};
    Types::State state{10, 10};
    Types::Model model{0};
    const Types::Search search{};
    Types::MatrixNode root{};

    const size_t iterations = 1 << 10;
    const size_t duration_ms = search.run_for_iterations(iterations, device, state, model, root);

    std::cout << iterations << " MCTS iterations completed in " << duration_ms << " ms." << std::endl;

    return 0;
}
