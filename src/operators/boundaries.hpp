#pragma once
#include <array>
#include <optional>
#include <vector>

#include "index_extents.hpp"

#include <sol/forward.hpp>

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

    friend auto operator<=>(const Line&, const Line&) = default;
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
constexpr auto df = Line{type::D, type::F};
constexpr auto fd = Line{type::F, type::D};
constexpr auto fn = Line{type::F, type::N};
constexpr auto nf = Line{type::N, type::F};

extern std::optional<std::pair<Grid, Object>> from_lua(const sol::table&, index_extents);

} // namespace ccs::bcs
