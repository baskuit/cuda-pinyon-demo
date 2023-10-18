#pragma once

#include <pinyon.hh>
#include <pkmn.h>

#include "./string-data.hh"

// global lambda for state processing

auto get_index_from_action = [](const uint8_t *bytes, const uint8_t choice, const uint8_t col_offset = 0)
{
    uint8_t type = choice & 3;
    uint8_t data = choice >> 2;
    if (type == 1)
    {
        uint8_t moveid = bytes[2 * (data - 1) + 168 + col_offset]; // 0 - 165
        return int64_t{moveid};                                    // index=0 is dummy very negative logit
    }
    else if (type == 2)
    {
        uint8_t slot = bytes[176 + data - 1 + col_offset];
        int dex = bytes[24 * (slot - 1) + 21 + col_offset]; // 0 - 151
        return int64_t{dex + 165};
    }
    else
    {
        return int64_t{0};
    }
};

/*

Pinyon compliant wrapper around the libpkmn battle objects.

The 3 functions get_actions, apply_actions, and randomize_transition are all that's needed for MCTS.

The chance/calc options allow us to force outcomes and thus reduce the branching factor of the search,
at the cost of fidelity.

*/

using TypeList = DefaultTypes<
    float,
    pkmn_choice,
    std::array<uint8_t, log_size>,
    bool,
    ConstantSum<1, 1>::Value,
    A<9>::Array>;

struct BattleTypes : TypeList
{

    class State : public PerfectInfoState<TypeList>
    {
    public:
        pkmn_gen1_battle battle;
        pkmn_gen1_log_options log_options;
        pkmn_gen1_battle_options options{};
        pkmn_result result{};
        pkmn_gen1_calc_options calc_options{}; // contains 24 byte override
        pkmn_gen1_chance_options chance_options{}; // rational, chance actions
        bool clamp = false;

        State(const int row_idx = 0, const int col_idx = 0)
        {
            const auto row_side = sides[row_idx];
            const auto col_side = sides[col_idx];
            memcpy(battle.bytes, row_side, 3 * 24);
            memcpy(battle.bytes + 144, row_side + 144, 40);
            memcpy(battle.bytes + 184, col_side, 3 * 24);
            memcpy(battle.bytes + 184 + 144, col_side + 144, 40);
            // memcpy(battle.bytes, row_side, 184);
            // memcpy(battle.bytes + 184, col_side, 184);

            for (int i = 2 * 184; i < n_bytes_battle; ++i)
            {
                battle.bytes[i] = 0;
            }
            log_options = {this->obs.get().data(), log_size};
            pkmn_gen1_battle_options_set(&options, &log_options, &chance_options, &calc_options);
            get_actions();
        }

        State(const State &other)
        {
            this->row_actions = other.row_actions;
            this->col_actions = other.col_actions;
            this->terminal = other.terminal;
            // memcpy(battle.bytes, other.battle.bytes, 384 - 8); // don't need seed
            memcpy(battle.bytes, other.battle.bytes, 384);
            log_options = {this->obs.get().data(), log_size};
            pkmn_gen1_battle_options_set(&options, &log_options, &chance_options, &calc_options);
            clamp = other.clamp;
        }

        State &operator=(const State &other)
        {
            this->row_actions = other.row_actions;
            this->col_actions = other.col_actions;
            this->terminal = other.terminal;
            memcpy(battle.bytes, other.battle.bytes, 384);
            log_options = {this->obs.get().data(), log_size};
            pkmn_gen1_battle_options_set(&options, &log_options, &chance_options, &calc_options);
            clamp = other.clamp;
            return *this;
        }

        void get_actions()
        {
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
            TypeList::Action col_action)
        {
            static const uint8_t rolls[2] = {217, 255};
            if (clamp)
            {
                calc_options.overrides.bytes[0] = rolls[battle.bytes[383] && 1];
                calc_options.overrides.bytes[8] = rolls[battle.bytes[382] && 1];
            }
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
                pkmn_gen1_battle_options_set(&options, &log_options, &chance_options, &calc_options);
            }
        }

        void randomize_transition(TypeList::PRNG &device)
        {
            uint8_t *battle_prng_bytes = battle.bytes + n_bytes_battle;
            *(reinterpret_cast<uint64_t *>(battle_prng_bytes)) = device.uniform_64();
        }

        // not pinyon interface, but reasonable to store here to unclog main.cc
        template <typename Buffers>
        void copy_to_buffer(
            Buffers buffers,
            const size_t buffer_index,
            const int rows, const int cols)
        {
            memcpy(
                &buffers.raw_input_buffer[buffer_index * 47],
                reinterpret_cast<const uint64_t *>(battle.bytes),
                376);
            for (int i = rows; i < 9; ++i)
            {
                buffers.joined_policy_buffer[buffer_index * 18 + i] = 0.0f;
            }

            for (int i = cols; i < 9; ++i)
            {
                buffers.joined_policy_buffer[buffer_index * 18 + 9 + i] = 0.0f;
            }
            for (int i = 0; i < 9; ++i)
            {
                if (i < rows)
                {
                    buffers.joined_policy_index_buffer[buffer_index * 18 + i] = get_index_from_action(battle.bytes, this->row_actions[i].get(), 0);
                }
                else
                {
                    buffers.joined_policy_index_buffer[buffer_index * 18 + i] = 0;
                }
            }
            if (buffers.joined_policy_index_buffer[buffer_index * 18] == 0 && rows == 1)
            {
                buffers.joined_policy_index_buffer[buffer_index * 18] = 1;
            }
            for (int i = 0; i < 9; ++i)
            {
                if (i < cols)
                {
                    buffers.joined_policy_index_buffer[buffer_index * 18 + 9 + i] = get_index_from_action(battle.bytes, this->col_actions[i].get(), 184);
                }
                else
                {
                    buffers.joined_policy_index_buffer[buffer_index * 18 + 9 + i] = 0;
                }
            }
            if (buffers.joined_policy_index_buffer[buffer_index * 18 + 9] == 0 && cols == 1)
            {
                buffers.joined_policy_index_buffer[buffer_index * 18 + 9] = 1;
            }
        }

        void print() const
        {
            std::cout << "ACTIONS: " << std::endl;
            for (const auto action : this->row_actions)
            {
                uint8_t data = action.get();
                std::string s = id_strings[get_index_from_action(this->battle.bytes, data, 0)];
                std::cout << s << ", ";
            }
            std::cout << std::endl;
            for (const auto action : this->col_actions)
            {
                uint8_t data = action.get();
                std::string s = id_strings[get_index_from_action(this->battle.bytes, data, 184)];
                std::cout << s << ", ";
            }
            std::cout << std::endl;
        }
    };
};
