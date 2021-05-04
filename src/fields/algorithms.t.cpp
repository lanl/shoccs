#include "scalar.hpp"
#include "tuple.hpp"
#include "vector.hpp"

#include "selector.hpp"

#include "types.hpp"

#include "algorithms.hpp"

#include <catch2/catch_test_macros.hpp>

#include <range/v3/algorithm/equal.hpp>

#include <vector>

#include <iostream>

using namespace ccs;

TEST_CASE("dot product")
{
    auto a = vector<std::vector<int>>{
        tuple{tuple{std::vector{1, 2, 3}},
              tuple{std::vector{1}, std::vector{2}, std::vector{3, 4}}},
        tuple{tuple{std::vector{4, 5, 6}},
              tuple{std::vector{2}, std::vector{3}, std::vector{4, 5}}},
        tuple{tuple{std::vector{7, 8, 9}},
              tuple{std::vector{3}, std::vector{4}, std::vector{5, 6}}}};
    vector<std::vector<int>> b = a;
    b *= 10;

    scalar<std::vector<int>> s = dot(a, b);

    REQUIRE(rs::equal(s | sel::D,
                      std::vector{1 * 10 + 4 * 40 + 7 * 70,
                                  2 * 20 + 5 * 50 + 8 * 80,
                                  3 * 30 + 6 * 60 + 9 * 90}));

    REQUIRE(rs::equal(s | sel::Rx, std::vector{10 + 2 * 20 + 3 * 30}));
    REQUIRE(rs::equal(s | sel::Ry, std::vector{2 * 20 + 3 * 30 + 4 * 40}));
    REQUIRE(rs::equal(s | sel::Rz,
                      std::vector{3 * 30 + 4 * 40 + 5 * 50, 4 * 40 + 5 * 50 + 6 * 60}));
};
