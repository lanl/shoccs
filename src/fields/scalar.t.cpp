#include "types.hpp"

#include "Scalar.hpp"
#include "Selector.hpp"

#include "lift.hpp"

#include <catch2/catch_test_macros.hpp>
#include <range/v3/algorithm/equal.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip_with.hpp>

TEST_CASE("construction")
{
    using namespace ccs::field;
    using T = std::vector<int>;
    auto s = Scalar<T>{};
}

TEST_CASE("selection")
{
    using namespace ccs;
    // some initialization
    auto v = std::vector{1, 2};
    auto rx = std::vector{3, 4};
    auto ry = std::vector{5, 6};
    auto rz = std::vector{7, 8};
    auto s = field::Scalar<std::vector<int>&>{field::Tuple{v}, field::Tuple{rx, ry, rz}};

    static_assert(std::same_as<decltype(s.get<0>()), field::Tuple<std::vector<int>&>&>);
    static_assert(std::same_as<decltype(view<0>(s.get<0>())), decltype(vs::all(v))&>);

    // Add tests to verify the type of the containers and views.
    // THey may point to scalar needing to be a tuple of nested tuples
    REQUIRE(rs::equal(v, s | selector::D));
    REQUIRE(rs::equal(rx, s | selector::Rx));
    REQUIRE(rs::equal(ry, s | selector::Ry));
    REQUIRE(rs::equal(rz, s | selector::Rz));

    static_assert(field::tuple::All<decltype(s | selector::D)>);

    auto t = s | selector::D;
    REQUIRE(rs::equal(t | vs::transform([](auto&& x) { return x * x * x; }),
                      std::vector{1, 8}));
}

TEST_CASE("math")
{
    using namespace ccs;
    using namespace ccs::field;
    auto s = Scalar<std::vector<int>>{
        Tuple{vs::iota(1, 5)}, Tuple{vs::iota(6, 10), vs::iota(6, 12), vs::iota(10, 15)}};

    auto q = s + 1;
    REQUIRE(rs::equal(vs::iota(2, 6), q | selector::D));
    REQUIRE(rs::equal(vs::iota(7, 11), q | selector::Rx));
    REQUIRE(rs::equal(vs::iota(7, 13), q | selector::Ry));
    REQUIRE(rs::equal(vs::iota(11, 16), q | selector::Rz));

    auto r = q + s;
    constexpr auto plus = [](auto&& a, auto&& b) {
        return vs::zip_with(std::plus{}, FWD(a), FWD(b));
    };
    REQUIRE(rs::equal(plus(vs::iota(2, 6), vs::iota(1, 5)), r | selector::D));
    REQUIRE(rs::equal(plus(vs::iota(11, 16), vs::iota(10, 15)), r | selector::Rz));

    Scalar<std::vector<int>> t = r;
    REQUIRE(rs::equal(t | selector::D, r | selector::D));
    REQUIRE(rs::equal(t | selector::Ry, r | selector::Ry));
}

TEST_CASE("lifting")
{
    using namespace ccs;
    using namespace ccs::field;
    auto s = Scalar<std::vector<int>>{
        Tuple{vs::iota(1, 5)}, Tuple{vs::iota(6, 10), vs::iota(6, 12), vs::iota(10, 15)}};

    constexpr auto f = lift([](auto&& x) { return std::abs(x) + 1; });

    auto j = f(s);
    auto k = s + 1;

    REQUIRE(rs::equal(j | selector::D, k | selector::D));
    REQUIRE(rs::equal(j | selector::Rx, k | selector::Rx));
    REQUIRE(rs::equal(j | selector::Ry, k | selector::Ry));
    REQUIRE(rs::equal(j | selector::Rz, k | selector::Rz));
}