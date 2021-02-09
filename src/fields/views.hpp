#pragma once

#include "types.hpp"
#include <range/v3/view/empty.hpp>

namespace ccs
{

namespace mesh
{

struct location_fn : rs::empty_view<real3> {
};

inline constexpr auto location = location_fn{};

} // namespace mesh

} // namespace ccs