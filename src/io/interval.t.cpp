#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "interval.hpp"

TEST_CASE("Never fire")
{
    auto it = ccs::interval<double>();
    REQUIRE(!it);
    for (int i = 0; i < 50; ++i) REQUIRE(!it(i));
}

TEST_CASE("No rollover")
{
    auto it = ccs::interval(1.5);
    for (int i = 0; i < 50; ++i) {
        double val = i * 0.5;
        if (i > 0 && i % 3 == 0) {
            REQUIRE(it(val, 1e-6));
            ++it;
        } else {
            REQUIRE(!it(val, 1e-6));
        }
    }
}

TEST_CASE("Rollover")
{
    auto it = ccs::interval(4.5);
    int fire_count = 0;
    for (int i = 0; i < 46; ++i) {
        auto ready = it(i, 1e-6);
        // Assert the first/last time it should fire
        if (i == 5) REQUIRE(ready);
        if (i == 45) REQUIRE(ready);
        if (ready) {
            ++fire_count;
            ++it;
        }
    }
    REQUIRE(fire_count == 10);
}
