#include "Scalar.hpp"
#include "Tuple.hpp"
#include "Vector.hpp"

#include "Selector.hpp"

#include "types.hpp"

#include "algorithms.hpp"

#include <catch2/catch_test_macros.hpp>

#include <range/v3/algorithm/equal.hpp>

#include <vector>

#include <iostream>

TEST_CASE("dot product")
{
    using namespace ccs;
    using namespace ccs::field;

    auto a = Vector<std::vector<int>>{
        Tuple{Tuple{std::vector{1, 2, 3}},
              Tuple{std::vector{1}, std::vector{2}, std::vector{3, 4}}},
        Tuple{Tuple{std::vector{4, 5, 6}},
              Tuple{std::vector{2}, std::vector{3}, std::vector{4, 5}}},
        Tuple{Tuple{std::vector{7, 8, 9}},
              Tuple{std::vector{3}, std::vector{4}, std::vector{5, 6}}}};
    Vector<std::vector<int>> b = a;
    b *= 10;

    Scalar<std::vector<int>> s = dot(a, b);

    REQUIRE(rs::equal(s | selector::D,
                      std::vector{1 * 10 + 4 * 40 + 7 * 70,
                                  2 * 20 + 5 * 50 + 8 * 80,
                                  3 * 30 + 6 * 60 + 9 * 90}));

    REQUIRE(rs::equal(s | selector::Rx, std::vector{10 + 2 * 20 + 3 * 30}));
    REQUIRE(rs::equal(s | selector::Ry, std::vector{2 * 20 + 3 * 30 + 4 * 40}));
    REQUIRE(rs::equal(s | selector::Rz,
                      std::vector{3 * 30 + 4 * 40 + 5 * 50, 4 * 40 + 5 * 50 + 6 * 60}));
};
