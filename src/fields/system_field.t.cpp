#include "system_field.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

using namespace ccs;

TEST_CASE("construction")
{
    // default construction
    auto x = field{};

    // sized construction
    auto y_size = system_size{2, 0, tuple{tuple{10}, tuple{2, 4, 5}}};
    auto y = field{y_size};

    auto&& [u, v] = y.scalars(0, 1);
    REQUIRE(rs::size(u | sel::D) == 10u);
    REQUIRE(rs::size(v | sel::D) == 10u);
    REQUIRE(rs::size(u | sel::Rx) == 2u);
    REQUIRE(rs::size(v | sel::Ry) == 4u);
    REQUIRE(rs::size(u | sel::Rz) == 5u);

    // boundary points in each direction

    // solid points ordered per each direction

    // auto q = field(3, int3{4, 5, 6}, boundary_points, solid_points);

    // auto [u, v] = q(0, 1);
}
