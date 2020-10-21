#include "result_r_tuple.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("crtp")
{
    using namespace ccs;
    q a{1};
    q b{2};

    q c = a + b;

    REQUIRE(c.value() == 3.f);
    REQUIRE((2 + q{1} + 2).value() == 5.f);
}