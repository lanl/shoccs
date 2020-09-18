#pragma once

#include <array>
#include <span>
#include <concepts>


namespace ccs
{
using real = double;
using real3 = std::array<real, 3>;
using real2 = std::array<real, 2>;
using int3 = std::array<int, 3>;
using int2 = std::array<int, 2>;

template<typename T>
using span = std::span<T>;

template <typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

}

