#pragma once

#include "index_extents.hpp"
#include "mesh/mesh_types.hpp"

#include <sol/forward.hpp>

namespace ccs
{
std::optional<std::pair<index_extents, domain_extents>>
extents_from_lua(const sol::table&);
}
