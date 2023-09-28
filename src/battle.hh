#include <pkmn.h>
#include <pinyon.hh>

#include "../sides.hh"

const size_t n_battle_bytes = 384;
const size_t n_battle_bytes_no_prng = 376;
// number of bytes from the log to use as the observation
const size_t log_size = 64;

const uint8_t n_pokemon = 151;
const uint8_t n_moveslots = 165;
const size_t policy_size = n_pokemon + n_moveslots;

struct BattleTypes;
