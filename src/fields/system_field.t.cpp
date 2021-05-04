#include "system_field.hpp"

TEST_CASE("construction")
{
    // default construction
    auto s = field{};

    // boundary points in each direction

    // solid points ordered per each direction

    auto q = field(3, int3{4, 5, 6}, boundary_points, solid_points);

    auto [u, v] = q(0, 1);
}

TEST_CASE("math") {}
