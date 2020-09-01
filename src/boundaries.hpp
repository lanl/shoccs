#pragma once

namespace ccs
{
enum class boundary { dirichlet, floating, neumann };

struct domain_boundaries {
    boundary left;
    boundary right;
};

}