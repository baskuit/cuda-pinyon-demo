# Efficient RL

This is a simple example of applying Pinyon to RL.

Here we:

* Wrap @pkmn/engine so that it's compatible with the library.
* Write a CUDA backend to quickly convert the battle bytes into tensor data
* Use the automatic `TreeBandit<Exp3<MonteCarloModel<BattleTypes>>>::Search` to easily define training workers
* Train a neural network on GPU using the training games

## pkmn/engine

RL is not possible without a very fast environment model, and that has notoriously been the case for Pokemon until the development of pkmn/engine.

# Setup

First clone the repo and its submodules: `engine`, `pinyon`, and `lrslib`.
Then build pkmn/engine and make sure its library file is in the directory specified in the `CMakeLists.txt`.
You will then likely have to comment out the tests and benchmark executables in the Pinyon `CmakeLIsts.txt`.
Otherwise you should be fine to just build `main`.

# Application

Ideally the model that is produced as a result of training is stronger than monte carlo evaluation at the same number of search iterations.
Unfortunately, it is not possible to use the model in the creation of more training data (AlphaZero). That requires a more complex batch inference backend.
This repo is hopefully a step towards that goal.
In the meantime, the trained network can still be wrapped as a Pinyon model and used in tree search without hardware acceleration.
