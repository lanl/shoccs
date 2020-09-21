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

TEST_CASE("selection")
{
    using namespace ccs;

    namespace rs = ranges;
    namespace vs = ranges::views;

    auto x =
        scalar_field<real, 0>{std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, int3{3, 2, 1}};
    scalar_field<real, 1> y = x;
    scalar_field<real, 2> z = x;

    vector_field v =
        scalar_field<real, 0>{std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, int3{3, 2, 1}};

    auto bi = std::vector{x.index({0, 0, 0}), x.index({0, 1, 0})};
    auto bj = std::vector{y.index({0, 0, 0}), y.index({0, 1, 0})};
    auto bk = std::vector{z.index({0, 0, 0}), z.index({0, 1, 0})};
    auto vfi = vector_field_index{bi, bj, bk};
    auto bvx = std::vector<real>{-1, -4};
    auto bvy = std::vector<real>{-2, -8};
    auto bvz = std::vector<real>{-4, -16};
    auto vfb = vector_field_bvalues{bvx, bvy, bvz};

    v(vfi) = vfb;

    x = v.x();
    REQUIRE(rs::equal(x, std::vector<real>{-1, 2, 3, -4, 5, 6}));
    x = v.y();
    REQUIRE(rs::equal(x, std::vector<real>{-2, 2, 3, -8, 5, 6}));
    x = v.z();
    REQUIRE(rs::equal(x, std::vector<real>{-4, 2, 3, -16, 5, 6}));
}