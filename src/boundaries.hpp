#pragma once
#include <array>
#include <vector>

namespace ccs
{
enum class boundary { dirichlet, floating, neumann };

struct domain_boundaries {
    boundary left;
    boundary right;
};

using grid_boundaries = std::array<domain_boundaries, 3>;
using object_boundaries = std::vector<boundary>;

}