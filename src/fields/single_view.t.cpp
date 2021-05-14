#include "container_tuple.hpp"
#include "view_tuple.hpp"

#include <catch2/catch_test_macros.hpp>

#include <range/v3/all.hpp>

using namespace ccs;

TEST_CASE("single_view")
{

    using T = std::vector<int>;
    T a{1, 2, 3};
    T b{4, 5, 6};

    auto u = single_view{view_tuple{a}};
    auto v = single_view{view_tuple{b}};

    REQUIRE(rs::equal(u, vs::iota(1, 4)));
    REQUIRE(rs::equal(v, vs::iota(4, 7)));

    u = v; // succeds with new assignment semantics
    REQUIRE(rs::equal(u, vs::iota(4, 7)));
    REQUIRE(rs::equal(v, vs::iota(4, 7)));
    REQUIRE(rs::equal(a, vs::iota(1, 4)));
}
