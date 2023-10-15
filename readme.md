
# Efficient NN Training

This is a simple example of applying Pinyon to train neural networks on Pokemon RBY. 

It is likely the fastest Pokemon AI setup (in terms of actor and learner `samples/sec`) in the world.

Here we:

* Wrap `pkmn/engine` so that its interface is compatible with the library.
* Write a CUDA back-end to handle processing and copying data between the actor, sample, and learner buffers.
* Use the automatically implemented search `TreeBandit<Exp3<MonteCarloModel<BattleTypes>>>::Search` to easily define MCTS-powered actors.
* Train multiple Libtorch models in parallel using the usual pythonic API
* Compare the strength of the models along their training histories with Pinyon's `Arena` utility, which uses the same bandit algorithms from tree search to optimize the model-evaluation process.

## Context

Pokemon is notoriously intractable domain for small-scale developers.

* No fast simulators

Reinforcement learning is not possible without a very fast environment model, something the community has sorely missed until the development of `pkmn/engine`. 
The most popular and previously and only simulator is Pokemon Showdown, which is designed for extensibility over performance. Unfortunately this NodeJS app is orders of magnitude slower than even novice implementations of games like Chess, Shogi, etc. This is due mostly to the complexity of Pokemon and its reliance on conditional-logic, but it's also due to PS's language and design choices.

* Unprincipled approaches and Scope

Because of the previous point, most researchers opted to use supervised learning approaches. The Showdown team will grant access to replay data if you ask, and this data was used in many policy gradient based attempts.

Imperfect information games like showdown are a well-known failure case for PG methods, at least prior to recent papers which claim that regularization can be used to ameliorate this.

Search based methods are one alternative, but search in imperfect information games is prohibitively expensive in the RL context.
That is why this demo and the first stage of Pinyon's development focuses on the *perfect information* context. This constraint is actually very natural and it allows us to use AlphaZero to train an agent. This algorithm is robust and requires minimal modification to work in the *simultaneous-move* regime.

To this end, Pinyon provides very fast and theoretically sound MCTS to generate the training data.


## AlphaZero

This repo is *nearly* an implementation of AlphaZero for single-machines, but it is lacking a CUDA back-end for using neural networks during the training. Emphasis on nearly because that is the most difficult thing to implement. This is my first CUDA app and hopefully its a step towards fast, open-source reinforcement learning.


# Setup

First clone the repo and its submodules: `engine`, `pinyon`, and `lrslib`.
Then build pkmn/engine and make sure its library file is in the directory specified in the `CMakeLists.txt`.
You will then likely have to comment out the tests and benchmark executables in the Pinyon `CmakeLIsts.txt`.
Otherwise you should be fine to just build `main`.

# Main

This section should be useful for understanding the project. It is an explanation of the data structures used in `main.cc`.

```cpp
namespace Options;
```
Contains the default values for net hyperparms, learning rate, search iterations, etc.

---

```cpp
using Types = TreeBandit<Exp3<MonteCarloModel<BattleTypes>>, DefaultNodes>;
```
Pinyon type list for Exp3 search on RBY battles using monte carlo evaluation.
`Types::PRNG` is an MT19937 engine.
`Types::State` is a battle, and takes the `sides` index for the row and column players in its constructor.
`Types::Model` is a Monte Carlo model and is initialized with a `uint64_t` seed.
`Types::Search` is the single-threaded MCTS with Exp3. Its `run_for_iterations`, `get_empirical_strategies`, `get_empirical_value` methods power the actors that generate the training data.

---

```cpp
struct Metric;

float Metric::update_and_get_rate(size_t count);
```
Simple mutex-guarded metric with an update function that returns the deposited samples/second.
One of the basic options is the max ratio between samples/second of the learners and the sample buffers. Basically, we don't want this ratio to be too high, as that means samples are being over-used. Since the actors are *significantly* slower than the learner on my machine, we throttle the latter.

---

```cpp
struct LearnerData
{
    const char cuda_device_index;
    const int batch_size;
    const float policy_loss_weight;
    const float base_learning_rate;
    const float log_learning_rate_decay;
    std::vector<std::string> saves{};
    Metric metric{Options::metric_history_size};
    size_t n_samples = 0;

    virtual W::Types::Model make_w_model() const = 0;
    virtual void step(LearnerBuffers, bool) = 0;
};
```
Contains more data, possibly. Base class that holds NN and optimizer independent data.
`make_w_model` produces a type erased `W::Types::Model` object that represents an agent in the `Arena` utility. This agent will perform a search for `Options::eval_iterations` with a CPU hosted version of the NN.
`step()` is learning on a single mini-batch. The derived class will have its own optimizer, but the data is the argument and the batch size/learning rate are in `LearnerData`. The second argument is a print flag.

 ---

```cpp
template <typename Net, typename Optimizer>
struct LearnerImpl : LearnerData {
    Net net;
    Optimizer optimizer;
};
```
Expects the `shared_ptr` module holder rather than the 'raw' class for `Net`. See the canonical `TORCH_MODULE` macro.


---

```cpp
using Learner = std::shared_ptr<LearnerData>;
```
Standard type-erasure so that the next class can hold different NN's and optimizers uniformly.

---

```cpp
struct TrainingAndEval {
	void run();
	void eval();
	void learn();
	void actor();
};
```
