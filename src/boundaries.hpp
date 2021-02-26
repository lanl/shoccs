#pragma once
#include <array>
#include <vector>

namespace ccs::bcs
{
enum class type {
    Dirichlet,
    D = Dirichlet,
    Floating,
    F = Floating,
    Neumann,
    N = Neumann
};

constexpr auto Dirichlet = type::Dirichlet;
constexpr auto Neumann = type::Neumann;
constexpr auto Floating = type::Floating;

struct Line {
    type left;
    type right;
};

using X = Line;
using Y = Line;
using Z = Line;

using Grid = std::array<Line, 3>;
using Object = std::vector<type>;

constexpr auto dd = Line{type::D, type::D};
constexpr auto ff = Line{type::F, type::F};
constexpr auto nn = Line{type::N, type::N};
constexpr auto dn = Line{type::D, type::N};
constexpr auto nd = Line{type::N, type::D};

} // namespace ccs::bcs