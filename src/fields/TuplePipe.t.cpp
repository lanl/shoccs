#include "Tuple.hpp"

#include "types.hpp"

#include "views.hpp"

#include <catch2/catch_test_macros.hpp>

#include <range/v3/algorithm/equal.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip_with.hpp>

#include <vector>

TEST_CASE("concepts")
{
    using namespace ccs;
    using namespace field::tuple;

    using T1 = Tuple<std::vector<real>>;
    using G = decltype([](auto&& i) { return i; });
    using F = decltype(vs::transform([](auto&& i) { return i; }));

    REQUIRE(traits::PipeableOver<F, T1>);
    REQUIRE(!traits::PipeableOver<G, T1>);

    using T3 = Tuple<std::span<int>, std::span<int>, std::span<const int>>;
    REQUIRE(traits::PipeableOver<F, T3>);
    REQUIRE(!traits::PipeableOver<G, T3>);

    using S = Tuple<T1, T3>;
    REQUIRE(traits::PipeableOver<F, S>);

    using V = Tuple<T3, T3>;
    REQUIRE(traits::PipeableOver<F, V>);
}

TEST_CASE("Pipe syntax for Owning OneTuples")
{
    using namespace ccs;
    using namespace field::tuple;

    auto x = Tuple{std::vector{1, 2, 3}};
    auto y = x | vs::transform([](auto&& i) { return i * i; });
    REQUIRE(rs::equal(y, std::vector{1, 4, 9}));

    auto z = Tuple<std::vector<int>>{};
    z = y | vs::transform([](auto&& i) { return i * i; });
    REQUIRE(rs::equal(z, std::vector{1, 16, 81}));
}

TEST_CASE("Pipe syntax for Non-Owning OneTuples")
{
    using namespace ccs;
    using namespace field::tuple;

    auto x = Tuple{std::vector{1, 2, 3}};
    auto y = x | vs::transform([](auto&& i) { return i * i; });
    REQUIRE(rs::equal(y, std::vector{1, 4, 9}));

    // copying should resize the vector when needed
    auto zz = std::vector<int>(3);

    auto z = Tuple{zz};
    z = y | vs::transform([](auto&& i) { return i * i; });
    REQUIRE(rs::size(zz) == rs::size(y));
    REQUIRE(rs::equal(zz, std::vector{1, 16, 81}));
    REQUIRE(rs::equal(z, zz));
}

TEST_CASE("Pipe syntax for ThreeTuples")
{
    using namespace ccs;
    using namespace field::tuple;

    auto x = Tuple{std::vector{1, 2, 3}, std::vector{4, 5}, std::vector{4, 3, 2, 1}};
    auto y = x | vs::transform([](auto&& i) { return i * i; });
    REQUIRE(traits::ThreeTuple<decltype(y)>);
    REQUIRE(rs::equal(get<0>(y), std::vector{1, 4, 9}));
    REQUIRE(rs::equal(get<1>(y), std::vector{16, 25}));
    REQUIRE(rs::equal(get<2>(y), std::vector{16, 9, 4, 1}));

    auto z = Tuple<std::vector<int>, std::vector<int>, std::vector<int>>{};
    z = y | vs::transform([](auto&& i) { return i + i; });
    REQUIRE(rs::equal(get<0>(z), std::vector{2, 8, 18}));
    REQUIRE(rs::equal(get<1>(z), std::vector{32, 50}));
    REQUIRE(rs::equal(get<2>(z), std::vector{32, 18, 8, 2}));
}

TEST_CASE("Pipe syntax for Non-Owning ThreeTuples")
{
    using namespace ccs;
    using namespace field::tuple;

    auto x = Tuple{std::vector{1, 2, 3}, std::vector{4, 5}, std::vector{4, 3, 2, 1}};
    auto y = x | vs::transform([](auto&& i) { return i * i; });
    REQUIRE(traits::ThreeTuple<decltype(y)>);
    REQUIRE(rs::equal(get<0>(y), std::vector{1, 4, 9}));
    REQUIRE(rs::equal(get<1>(y), std::vector{16, 25}));
    REQUIRE(rs::equal(get<2>(y), std::vector{16, 9, 4, 1}));

    auto a = std::vector<int>(3);
    auto b = std::vector<int>(2);
    auto c = std::vector<int>(4);
    auto z = Tuple{a, b, c};

    z = y | vs::transform([](auto&& i) { return i + i; });
    REQUIRE(rs::equal(get<0>(z), std::vector{2, 8, 18}));
    REQUIRE(rs::equal(get<1>(z), std::vector{32, 50}));
    REQUIRE(rs::equal(get<2>(z), std::vector{32, 18, 8, 2}));
}

TEST_CASE("ThreeTuples with ThreeTuplePipes")
{
    using namespace ccs;
    using namespace field::tuple;
    constexpr auto f = vs::transform([](auto&& i) { return i * i; });
    constexpr auto g = vs::transform([](auto&& i) { return i + i; });
    constexpr auto h = vs::transform([](auto&& i) { return i * i * i; });

    auto x = Tuple{std::vector{1, 2, 3}, std::vector{4, 5}, std::vector{4, 3, 2, 1}};
    auto y = x | std::tuple{f, g, h};
    REQUIRE(traits::ThreeTuple<decltype(y)>);
    REQUIRE(rs::equal(get<0>(y), std::vector{1, 4, 9}));
    REQUIRE(rs::equal(get<1>(y), std::vector{8, 10}));
    REQUIRE(rs::equal(get<2>(y), std::vector{64, 27, 8, 1}));

    auto a = std::vector<int>(3);
    auto b = std::vector<int>(2);
    auto c = std::vector<int>(4);
    auto z = Tuple{a, b, c};
    z = y | g; // vs::transform([](auto&& i) { return i + i; });
    REQUIRE(rs::equal(get<0>(z), std::vector{2, 8, 18}));
    REQUIRE(rs::equal(get<1>(z), std::vector{16, 20}));
    REQUIRE(rs::equal(get<2>(z), std::vector{128, 54, 16, 2}));

    auto q = Tuple<std::span<int>, std::span<int>, std::span<int>>{a, b, c};

    q | std::tuple{f, g, h};

    q = y | Tuple{f, g, h};

    REQUIRE(rs::equal(a, std::vector{1, 16, 81}));
    REQUIRE(rs::equal(b, std::vector{16, 20}));
    REQUIRE(rs::equal(c, std::vector{262144, 19683, 512, 1}));

    // vs::transform([](auto&& i) { return i; }) |
    //     Tuple{vs::transform([](auto&& i) { return i * i; }),
    //           vs::transform([](auto&& i) { return i + i; }),
    //           vs::transform([](auto&& i) { return i * i * i; })};
    // vs::transform([](auto&& i) { return i; }) |
    //     Tuple{vs::transform([](auto&& i) { return 2 * i; })};
}

TEST_CASE("Nested Pipe")
{
    using namespace ccs;
    using namespace field::tuple;

    auto x =
        Tuple{Tuple{std::vector{1, 2, 3}, std::vector{4, 5}, std::vector{4, 3, 2, 1}},
              Tuple{std::vector{-1, -2, -3}}};
    auto y = x | vs::transform([](auto&& i) { return i + 2; });

    REQUIRE(rs::equal(get<0, 0>(y), std::vector{3, 4, 5}));
    REQUIRE(rs::equal(get<1, 0>(y), std::vector{6, 7}));
    REQUIRE(rs::equal(get<2, 0>(y), std::vector{6, 5, 4, 3}));
    REQUIRE(rs::equal(get<0, 1>(y), std::vector{1, 0, -1}));
}
