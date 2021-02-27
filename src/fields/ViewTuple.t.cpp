#include "ViewTuple.hpp"

#include <catch2/catch_test_macros.hpp>

#include <range/v3/algorithm/equal.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/repeat_n.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/zip_with.hpp>

#include <vector>

#include <iostream>

TEST_CASE("construction")
{
    using namespace ccs;
    using namespace ccs::field::tuple;
    using T = std::vector<real>;

    auto v = T{1, 2, 3};
    ViewTuple<T&> x{v};

    REQUIRE(rs::equal(x, v));

    auto [y] = x;
    REQUIRE(rs::equal(y, v));

    auto vv = T{3, 4, 5};
    ViewTuple<T&, T&> z{v, vv};
    auto [a, b] = z;

    REQUIRE(rs::equal(a, std::vector<real>{1, 2, 3}));
    REQUIRE(rs::equal(b, std::vector<real>{3, 4, 5}));
    for (auto&& i : b) i *= 2;
    REQUIRE(rs::equal(vv, T{6, 8, 10}));
}

TEST_CASE("assignment")
{
    using namespace ccs;
    using namespace ccs::field::tuple;
    using T = std::vector<real>;

    auto v = T{1, 2, 3};
    ViewTuple<T&> x{v};

    x = -1;
    REQUIRE(rs::equal(v, T{-1, -1, -1}));
    REQUIRE(rs::equal(v, x));
    REQUIRE(rs::equal(get<0>(x), v));
}