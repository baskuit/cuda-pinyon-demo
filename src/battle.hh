#pragma once

#include <pkmn.h>
#include <pinyon.hh>

#include "buffers.hh"
#include "../sides.hh"

// number of bytes from the log to use as the observation
const size_t log_size = 64;

using BrokenTypeList = DefaultTypes<
    float,                         // floating point type used for calculations
    pkmn_choice,                   // action type for battles
    std::array<uint8_t, log_size>, // observation type after commiting actions
    bool,                          // transition probabilities are not return by this configuration of engine (see 'calc' and 'chance' flags)
    ConstantSum<1, 1>::Value,
    A<9>::Array>;

// shadow old vector type with libpkmn interface friendly type. the battles native 'get_actions' method expects a c style array.
// also shadow `TypeList` to fix the reflexive type declaration. Note: this declaration is not actually used in this example, but it's interface proper.
struct FixedTypeList : BrokenTypeList
{
    // using VectorAction = A<9>::Array<BrokenTypeList::Action>;
    // using TypeList = FixedTypeList;
    // TODO 64bit strategies for smaller matrix stats!. Instead of 32bit x 9
};

using TypeList = FixedTypeList;

struct BattleTypes : TypeList
{

    class State : public PerfectInfoState<TypeList>
    {
    public:
        // this wrapper mirrors `example.c` in engine repo
        pkmn_gen1_battle battle;
        pkmn_gen1_log_options log_options;
        pkmn_gen1_battle_options options{};
        pkmn_result result;

        State(const uint64_t seed = 0)
        {
            // pick random teams, copy to battle bytes
            // initialize the rest of the battle
            TypeList::PRNG device{seed};
            const auto row_side = sides[device.random_int(n_sides)];
            const auto col_side = sides[device.random_int(n_sides)];
            memcpy(battle.bytes, row_side, 184);
            memcpy(battle.bytes + 184, col_side, 184);
            for (int i = 2 * 184; i < n_bytes_battle; ++i)
            {
                battle.bytes[i] = 0;
            }
            log_options = {this->obs.get().data(), log_size};
            pkmn_gen1_battle_options_set(&options, &log_options, NULL, NULL);
            get_actions();
        }

        // we redefine the copy constructor so that it skips copying some helper stuff
        State(const State &other)
        {
            this->row_actions = other.row_actions;
            this->col_actions = other.col_actions;
            this->terminal = other.terminal;
            // this->obs = other.obs;
            // this->prob = other.prob;

            // memcpy(battle.bytes, other.battle.bytes, 384 - 8); // don't need seed
            memcpy(battle.bytes, other.battle.bytes, 384);
            // offset copy probably slower
            log_options = {this->obs.get().data(), log_size};
            pkmn_gen1_battle_options_set(&options, &log_options, NULL, NULL);
        }

        State& operator=(const State &other)
        {
            this->row_actions = other.row_actions;
            this->col_actions = other.col_actions;
            this->terminal = other.terminal;
            memcpy(battle.bytes, other.battle.bytes, 384);
            log_options = {this->obs.get().data(), log_size};
            pkmn_gen1_battle_options_set(&options, &log_options, NULL, NULL);
            return *this;
        }

        void get_actions()
        {
            // this is why we made the action vector type an array
            this->row_actions.resize(
                pkmn_gen1_battle_choices(
                    &battle,
                    PKMN_PLAYER_P1,
                    pkmn_result_p1(result),
                    reinterpret_cast<pkmn_choice *>(this->row_actions.data()),
                    PKMN_MAX_CHOICES));
            this->col_actions.resize(
                pkmn_gen1_battle_choices(
                    &battle,
                    PKMN_PLAYER_P2,
                    pkmn_result_p2(result),
                    reinterpret_cast<pkmn_choice *>(this->col_actions.data()),
                    PKMN_MAX_CHOICES));
        }

        void apply_actions(
            TypeList::Action row_action,
            TypeList::Action col_action) // the type given to DefaultTypes was uint8_t, but that type list gave it a wrapper for strong-typing
        // so we have to use Types::Action. In general, just use `Types::` + `interface standard type-identifier`
        {
            result = pkmn_gen1_battle_update(&battle, row_action.get(), col_action.get(), &options);
            const pkmn_result_kind result_kind = pkmn_result_type(result);
            if (result_kind) [[unlikely]]
            {
                this->terminal = true;
                if (result_kind == PKMN_RESULT_WIN)
                {
                    this->payoff = TypeList::Value{1.0f};
                }
                else if (result_kind == PKMN_RESULT_LOSE)
                {
                    this->payoff = TypeList::Value{0.0f};
                }
                else
                {
                    this->payoff = TypeList::Value{0.5f};
                }
            }
            else [[likely]]
            {
                pkmn_gen1_battle_options_set(&options, NULL, NULL, NULL);
            }
        }

        void randomize_transition(TypeList::PRNG &device)
        {
            // reinterpret start of prng as one 64 bit, assign at once
            uint8_t *battle_prng_bytes = battle.bytes + n_bytes_battle;
            *(reinterpret_cast<uint64_t *>(battle_prng_bytes)) = device.get_seed();
        }
    };
};
