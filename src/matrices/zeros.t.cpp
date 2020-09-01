#include "zeros.hpp"

#include <catch2/catch_test_macros.hpp>

#include <range/v3/view/all.hpp>

TEST_CASE("zeros")
{
    using namespace ccs;

    auto A = matrix::zeros(10);
    std::vector<real> x(12, 7.0);

    auto b = A * x;

    REQUIRE(b.size() == 10u);

    for (int i = 0; i < 10; i++) { REQUIRE(b[i] == 0.0); }
}