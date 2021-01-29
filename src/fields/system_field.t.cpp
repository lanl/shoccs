#include "SystemField.hpp"

TEST_CASE("construction") {
    // default construction
    auto s = SystemField{};

    // boundary points in each direction

    // solid points ordered per each direction

    auto q = SystemField(3, int3{4, 5, 6}, boundary_points, solid_points);

    auto [u, v] = q(0, 1);
}

TEST_CASE("math")
{

}