#include "vector_field.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <range/v3/algorithm/equal.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/repeat_n.hpp>

TEST_CASE("from scalar")
{
    using namespace ccs;

    namespace rs = ranges;
    namespace vs = ranges::views;

    auto x =
        scalar_field<real, 0>{std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, int3{3, 2, 1}};

    vector_field v =
        scalar_field<real, 0>{std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, int3{3, 2, 1}};

    auto vz = v.z();
    vz += v.y();
    vz += v.x();

    scalar_field<real, 0> xz = vz;

    REQUIRE(rs::equal(xz, x + x + x));


}