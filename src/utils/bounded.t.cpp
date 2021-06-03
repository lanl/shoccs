#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "bounded.hpp"

using namespace ccs;

TEST_CASE("bounded default")
{
    auto x = bounded<int>{};
    REQUIRE(!x);
    x++;
    ++x;
    REQUIRE(!x);
    x += -2;
    REQUIRE(!x);
    x += -5;
    REQUIRE(!x);
}

TEST_CASE("bounded")
{
    auto x = bounded<real>{0, 0, 4.3};
    REQUIRE(x);
    x += 2.2;
    REQUIRE(x);
    x += 2.2;
    REQUIRE(!x);

    auto y = bounded<real>{4.5};
    REQUIRE(y);
    y += 2.2;
    REQUIRE(y);
    y += 2.2;
    REQUIRE(y);
    ++y;
    REQUIRE(!y);
}
