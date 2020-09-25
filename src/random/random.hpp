#pragma once

#include "types.hpp"
#include <random>

namespace ccs
{
// helper functions for randomness from Walter Brown.
std::default_random_engine& global_urng();

void randomize();

int pick(int from, int thru);

real pick(real from = 0, real upto = 1);

} // namespace ccs