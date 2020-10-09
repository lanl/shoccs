#include "scalar_field.hpp"
#include "select.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <range/v3/algorithm/equal.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/repeat_n.hpp>
#include <range/v3/view/transform.hpp>

TEST_CASE("transposing")
{
    using namespace ccs;

    auto x =
        scalar_field<real, 0>{std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, int3{3, 2, 1}};
    REQUIRE(x.size() == 6u);
    auto y = scalar_field<real, 0>{std::vector{-1.0, -2.0, -3.0, -4.0, -5.0, -6.0},
                                   int3{3, 2, 1}};
    REQUIRE(y.size() == 6u);
    auto sum = x + y + 2 * x - y + 3 * y;

    REQUIRE(rs::equal(sum, vs::repeat_n(0.0, sum.size())));

    // construction
    auto z = scalar_field{sum};
    REQUIRE(rs::equal(sum, z));
    // assignment
    auto zz = scalar_field{};
    zz = sum;
    REQUIRE(rs::equal(sum, zz));

    z = x + x;
    REQUIRE(rs::equal(z, std::vector<real>{2, 8, 4, 10, 6, 12}));

    scalar_field<real, 1> q = z;

    REQUIRE(rs::equal(z, q));
}

TEST_CASE("3d")
{
    using namespace ccs;

    auto v = vs::iota(1, 13) | rs::to<std::vector<real>>();

    scalar_field<real, 0> x{std::move(v), int3{3, 2, 2}};
    scalar_field<real, 1> y{x};
    scalar_field<real, 2> z{y};

    // test some identity transformations
    {
        scalar_field<real, 0> xx{y};
        REQUIRE(rs::equal(x, xx));
        xx = z;
        REQUIRE(rs::equal(x, xx));
    }

    {
        scalar_field<real, 1> yy{z};
        REQUIRE(rs::equal(y, yy));
        yy = x;
        REQUIRE(rs::equal(y, yy));
    }

    {
        scalar_field<real, 2> zz{x};
        REQUIRE(rs::equal(z, zz));
        zz = y;
        REQUIRE(rs::equal(z, zz));
    }

    {
        auto xx = x;
        xx += y;
        xx += z;
        auto x3 = 3 * x;
        REQUIRE(rs::equal(xx, x3));
    }
}

TEST_CASE("selection")
{
    using namespace ccs;

    auto x = scalar_field<real, 0>{std::vector<real>{1, 2, 3, 4, 5, 6}, int3{3, 2, 1}};

    auto ind = std::vector<int>{0, 2, 5};

    x >> select(ind) <<= std::vector<real>{-1, -3, -6};

    REQUIRE(rs::equal(x, std::vector<real>{-1, 2, -3, 4, 5, -6}));

    std::vector<real> w{};
    w <<= x >> select(ind) >> vs::transform([](auto&& v) { return v * v; });

    REQUIRE(w == std::vector<real>{1, 9, 36});

    // these coordinates are in ijk and thus do not depend on the orientation of the
    // scalar field
    const auto i = std::vector<int3>{{0, 0, 0}, {2, 0, 0}, {2, 1, 0}};

    scalar_field<real, 2> z = x;
    z >> select(i) <<= std::vector<real>{1, 3, 6};
    scalar_field<real, 0> xx = z;

    REQUIRE(rs::equal(xx, std::vector<real>{1, 2, 3, 4, 5, 6}));

    REQUIRE(rs::equal(xx | vs::transform([](auto&& v) { return v * v; }),
                      std::vector<real>{1, 4, 9, 16, 25, 36}));

    auto xxt = xx >> vs::transform([](auto&& v) { return v * v; });
    REQUIRE(rs::equal(xxt, std::vector<real>{1, 4, 9, 16, 25, 36}));
    REQUIRE(Scalar_Field<decltype(xxt)>);
    REQUIRE(Compatible_Fields<decltype(xx), decltype(xxt)>);
}

TEST_CASE("from input range")
{
    using namespace ccs;

    auto z = z_field(int3{4, 5, 6});
    z = vs::iota(0);

    REQUIRE(rs::equal(z, vs::iota(0) | vs::take(120)));
}