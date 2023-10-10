#pragma once

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
        pkmn_result result;
        pkmn_gen1_calc_options calc_options{}; // contains 24 byte override
        pkmn_gen1_chance_options chance_options{};

        State(const int row_idx = 0, const int col_idx = 0)
        {
            const auto row_side = sides[row_idx];
            const auto col_side = sides[col_idx];
            memcpy(battle.bytes, row_side, 184);
            memcpy(battle.bytes + 184, col_side, 184);
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
        }

        State &operator=(const State &other)
        {
            this->row_actions = other.row_actions;
            this->col_actions = other.col_actions;
            this->terminal = other.terminal;
            memcpy(battle.bytes, other.battle.bytes, 384);
            log_options = {this->obs.get().data(), log_size};
            pkmn_gen1_battle_options_set(&options, &log_options, &chance_options, &calc_options);
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
            const uint8_t roll{255};
            memset(calc_options.overrides.bytes, roll, 8);
            memset(calc_options.overrides.bytes + 8, roll, 8);
            pkmn_gen1_battle_options_set(&options, &log_options, &chance_options, &calc_options);
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
            uint8_t *battle_prng_bytes = battle.bytes + n_bytes_battle;
            *(reinterpret_cast<uint64_t *>(battle_prng_bytes)) = device.uniform_64();
        }
    };
};

// global lambda for state processing

auto get_index_from_action = [](const uint8_t *bytes, const uint8_t choice, const uint8_t col_offset = 0)
{
    uint8_t type = choice & 3;
    uint8_t data = choice >> 2;
    if (type == 1)
    {
        uint8_t moveid = bytes[2 * (data - 1) + 10 + col_offset]; // 0 - 165
        return int64_t{moveid};                                   // index=0 is dummy very negative logit
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
