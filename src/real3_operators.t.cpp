#include "real3_operators.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "std_matchers.hpp"

TEST_CASE("real3/real3")
{
    using namespace ccs;
    using Catch::Matchers::Approx;

    real3 a = {1, 2, 3};
    real3 b = {4, 5, 6};

    REQUIRE(a + b == real3{5, 7, 9});
    REQUIRE(a - b == real3{-3, -3, -3});
    REQUIRE_THAT(b / a, Approx(real3{4, 2.5, 2}));
    REQUIRE(b * a == a * b);
}

TEST_CASE("real3/Numeric")
{
    using namespace ccs;
    using Catch::Matchers::Approx;

    real3 a = {1, 2, 3};

    REQUIRE(a + 0 == a);
    REQUIRE(0 - a == -1 * a);
    REQUIRE(-1 * a == a * -1);
    REQUIRE_THAT(a / 2, Approx(real3{0.5, 1, 3.0 / 2}));
}