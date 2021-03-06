#pragma once

#include <array>

namespace ccs
{

using real = double;
using real3 = std::array<real, 3>;
using real2 = std::array<real, 2>;

using integer = long; // prefer higher precision than regular int
using int3 = std::array<int, 3>;
using int2 = std::array<int, 2>;
} // namespace ccs
