
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
