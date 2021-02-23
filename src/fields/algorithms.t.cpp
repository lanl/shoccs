#include "Scalar.hpp"
#include "Tuple.hpp"
#include "Vector.hpp"

#include "Selector.hpp"
#include "views.hpp"

#include "types.hpp"

#include "algorithms.hpp"

#include <catch2/catch_test_macros.hpp>

#include <range/v3/algorithm/equal.hpp>

#include <vector>

#include <iostream>

// construct simple mesh geometry
namespace global
{

const auto x = std::vector<ccs::real>{0, 1, 2, 3, 4};
const auto y = std::vector<ccs::real>{-2, -1};
const auto z = std::vector<ccs::real>{6, 7, 8, 9};
const auto rx = std::vector<ccs::real3>{{0.5, -2, 6}, {1.5, -1, 9}};
const auto ry = std::vector<ccs::real3>{{1, -1.75, 7}, {4, -1.25, 7}, {3, -1.1, 9}};
const auto rz = std::vector<ccs::real3>{{0, -2, 6.1}};
const auto loc = ccs::mesh::Location{x, y, z, rx, ry, rz};

} // namespace global

TEST_CASE("ThreeTuple to OneTuple via reduce")
{
    using namespace ccs;
    using namespace ccs::field;

    auto x = Tuple{std::vector{1, 2, 3}, std::vector{4, 5, 6}, std::vector{4, 3, 2}};

    auto yData = std::vector<int>(3);
    auto y = Tuple<std::span<int>>{yData};

    y = reduce(x, [](auto&& x, auto&& y, auto&& z) { return x * x + y * y + z * z; });
    REQUIRE(rs::equal(y,
                      std::vector{1 * 1 + 4 * 4 + 4 * 4,
                                  2 * 2 + 5 * 5 + 3 * 3,
                                  3 * 3 + 6 * 6 + 2 * 2}));
}

TEST_CASE("dot product")
{
    using namespace ccs;
    using namespace ccs::field;

    auto s = SimpleScalar<std::vector<int>>{};

    auto a = SimpleVector<std::vector<int>>{
        &global::loc,
        Tuple{std::vector{1, 2, 3}, std::vector{4, 5, 6}, std::vector{7, 8, 9}},
        Tuple{std::vector{1}, std::vector{2}, std::vector{3}}};
    auto b = SimpleVector<std::vector<int>>{
        &global::loc,
        Tuple{std::vector{10, 20, 30}, std::vector{40, 50, 60}, std::vector{70, 80, 90}},
        Tuple{std::vector{10}, std::vector{20}, std::vector{30}}};

    s = dot(a, b);

    REQUIRE(rs::equal(s | selector::D,
                      std::vector{1 * 10 + 4 * 40 + 7 * 70,
                                  2 * 20 + 5 * 50 + 8 * 80,
                                  3 * 30 + 6 * 60 + 9 * 90}));

    REQUIRE(rs::equal(s | selector::Rx, std::vector{10}));
    REQUIRE(rs::equal(s | selector::Ry, std::vector{40}));
    REQUIRE(rs::equal(s | selector::Rz, std::vector{90}));

    auto x = std::vector<int>(3);
    auto rx = std::vector<int>(1);
    auto ry = std::vector<int>(1);
    auto rz = std::vector<int>(1);
    using T = std::span<int>;
    auto ss = SimpleScalar<T>{&global::loc, Tuple{T(x)}, Tuple{T(rx), T(ry), T(rz)}};

    ss = dot(a, b);
    REQUIRE(rs::equal(s | selector::D, x));
    REQUIRE(rs::equal(s | selector::Rx, rx));
    REQUIRE(rs::equal(s | selector::Ry, ry));
    REQUIRE(rs::equal(s | selector::Rz, rz));
};

// could we write the dot product in this way?
// a* b | vs::transform([](auto&& x, auto&& y, auto&& z) { return x + y + z; });
// maybe not..
// D = Dx * Dx + Dy * Dy + Dz * Dz
// Rx = Rx * Rx
// Ry = Ry * Ry
// Rz = Rz * Rz
// auto ab = a * b;
// s | selector::D =
// ab | selector::D |
//    vs::transform([](auto&& x, auto&& y, auto&& z) { return x + y + z; });
// s | selector::Rxyz = ab | selector::Rxyz;
