#pragma once

#include "Location.hpp"
#include "types.hpp"
#include <range/v3/view/empty.hpp>
#include <range/v3/view/repeat_n.hpp>
#include <range/v3/view/view.hpp>

namespace ccs
{

namespace mesh
{

struct location_fn : rs::empty_view<real3> {
};

inline constexpr auto location =
    rs::make_view_closure([](auto&& selection) { return selection.location; });

} // namespace mesh

} // namespace ccs