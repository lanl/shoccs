#include "scalar.hpp"

#include "matchers.hpp"
#include <catch2/catch_test_macros.hpp>

#include <range/v3/view/iota.hpp>

TEST_CASE("initialization")
{
    using namespace ccs;
    using Catch::Matchers::Approx;
    using T = std::vector<real>;

    // some intialization
    auto s = x_scalar{x_field{vs::iota(0), int3{4, 5, 6}},
                      vector_range{T{0, 1, 2}, T{4, 5}, T{6, 7, 8}}};
    auto s1 = x_scalar{x_field{vs::iota(1), int3{4, 5, 6}},
                       vector_range{T{1, 2, 3}, T{5, 6}, T{7, 8, 9}}};

    s += 1;
    REQUIRE_THAT(s, Approx(s1));

    s = 0;
    REQUIRE_THAT(s, Approx(x_scalar{int3{4, 5, 6}, int3{3, 2, 3}}));

    s += s1 * 1;
    REQUIRE_THAT(s, Approx(s1));

    s = s1 * 2 - s1;
    REQUIRE_THAT(s, Approx(s1));

    s += 3 * s1 - 2 * s1 - s1;
    REQUIRE_THAT(s, Approx(s1));
}

#if 0
TEST_CASE("selection")
{
    // this should also set the relevant solid points in the scalar
    s.obj_boundaries = geom.Rxyz() >> vs::transform(f);

    // update 
    s0 += ds * dt;
}
#endif