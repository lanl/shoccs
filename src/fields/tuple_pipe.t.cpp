#include "tuple.hpp"

#include "types.hpp"

#include <catch2/catch_test_macros.hpp>

#include <range/v3/algorithm/equal.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip_with.hpp>

#include <vector>

using namespace ccs;

TEST_CASE("concepts")
{
    using T1 = tuple<std::vector<real>>;
    using G = decltype([](auto&& i) { return i; });
    using F = decltype(vs::transform([](auto&& i) { return i; }));

    REQUIRE(PipeableOver<F, T1>);
    REQUIRE(!PipeableOver<G, T1>);

    using T3 = tuple<std::span<int>, std::span<int>, std::span<const int>>;
    REQUIRE(PipeableOver<F, T3>);
    REQUIRE(!PipeableOver<G, T3>);

    using S = tuple<T1, T3>;
    REQUIRE(PipeableOver<F, S>);

    using V = tuple<T3, T3>;
    REQUIRE(PipeableOver<F, V>);
}

TEST_CASE("Pipe syntax for Owning OneTuples")
{
    auto x = tuple{std::vector{1, 2, 3}};
    auto y = x | vs::transform([](auto&& i) { return i * i; });
    REQUIRE(rs::equal(y, std::vector{1, 4, 9}));

    auto z = tuple<std::vector<int>>{};
    z = y | vs::transform([](auto&& i) { return i * i; });
    REQUIRE(rs::equal(z, std::vector{1, 16, 81}));
}

TEST_CASE("Pipe syntax for Non-Owning OneTuples")
{
    auto x = tuple{std::vector{1, 2, 3}};
    auto y = x | vs::transform([](auto&& i) { return i * i; });
    REQUIRE(rs::equal(y, std::vector{1, 4, 9}));

    // copying should resize the vector when needed
    auto zz = std::vector<int>(3);

    auto z = tuple{zz};
    z = y | vs::transform([](auto&& i) { return i * i; });
    REQUIRE(rs::size(zz) == rs::size(y));
    REQUIRE(rs::equal(zz, std::vector{1, 16, 81}));
    REQUIRE(rs::equal(z, zz));
}

TEST_CASE("Pipe syntax for ThreeTuples")
{
    auto x = tuple{std::vector{1, 2, 3}, std::vector{4, 5}, std::vector{4, 3, 2, 1}};
    auto y = x | vs::transform([](auto&& i) { return i * i; });
    REQUIRE(ThreeTuple<decltype(y)>);
    REQUIRE(y ==
            tuple{std::vector{1, 4, 9}, std::vector{16, 25}, std::vector{16, 9, 4, 1}});

    auto z = tuple<std::vector<int>, std::vector<int>, std::vector<int>>{};
    z = y | vs::transform([](auto&& i) { return i + i; });
    REQUIRE(z ==
            tuple{std::vector{2, 8, 18}, std::vector{32, 50}, std::vector{32, 18, 8, 2}});
}

TEST_CASE("Pipe syntax for Non-Owning ThreeTuples")
{
    auto x = tuple{std::vector{1, 2, 3}, std::vector{4, 5}, std::vector{4, 3, 2, 1}};
    auto y = x | vs::transform([](auto&& i) { return i * i; });
    REQUIRE(ThreeTuple<decltype(y)>);
    REQUIRE(y ==
            tuple{std::vector{1, 4, 9}, std::vector{16, 25}, std::vector{16, 9, 4, 1}});

    auto a = std::vector<int>(3);
    auto b = std::vector<int>(2);
    auto c = std::vector<int>(4);
    auto z = tuple{a, b, c};

    z = y | vs::transform([](auto&& i) { return i + i; });
    REQUIRE(z ==
            tuple{std::vector{2, 8, 18}, std::vector{32, 50}, std::vector{32, 18, 8, 2}});
}

TEST_CASE("ThreeTuples with ThreeTuplePipes")
{
    constexpr auto f = vs::transform([](auto&& i) { return i * i; });
    constexpr auto g = vs::transform([](auto&& i) { return i + i; });
    constexpr auto h = vs::transform([](auto&& i) { return i * i * i; });

    auto x = tuple{std::vector{1, 2, 3}, std::vector{4, 5}, std::vector{4, 3, 2, 1}};
    auto y = x | std::tuple{f, g, h};
    REQUIRE(ThreeTuple<decltype(y)>);
    REQUIRE(y ==
            tuple{std::vector{1, 4, 9}, std::vector{8, 10}, std::vector{64, 27, 8, 1}});

    auto a = std::vector<int>(3);
    auto b = std::vector<int>(2);
    auto c = std::vector<int>(4);
    auto z = tuple{a, b, c};
    z = y | g; // vs::transform([](auto&& i) { return i + i; });
    REQUIRE(z == tuple{std::vector{2, 8, 18},
                       std::vector{16, 20},
                       std::vector{128, 54, 16, 2}});

    auto q = tuple<std::span<int>, std::span<int>, std::span<int>>{a, b, c};

    q | std::tuple{f, g, h};

    q = y | tuple{f, g, h};

    REQUIRE(rs::equal(a, std::vector{1, 16, 81}));
    REQUIRE(rs::equal(b, std::vector{16, 20}));
    REQUIRE(rs::equal(c, std::vector{262144, 19683, 512, 1}));

    vs::transform([](auto&& i) { return i; }) |
        tuple{vs::transform([](auto&& i) { return i * i; }),
              vs::transform([](auto&& i) { return i + i; }),
              vs::transform([](auto&& i) { return i * i * i; })};
    // vs::transform([](auto&& i) { return i; }) |
    //     tuple{vs::transform([](auto&& i) { return 2 * i; })};
}

TEST_CASE("Nested Pipe")
{
    auto x =
        tuple{tuple{std::vector{1, 2, 3}, std::vector{4, 5}, std::vector{4, 3, 2, 1}},
              tuple{std::vector{-1, -2, -3}}};
    auto y = x | vs::transform([](auto&& i) { return i + 2; });

    REQUIRE(y ==
            tuple{tuple{std::vector{3, 4, 5}, std::vector{6, 7}, std::vector{6, 5, 4, 3}},
                  tuple{std::vector{1, 0, -1}}});
}

TEST_CASE("constexpr tuples of transforms")
{
    constexpr auto tup = tuple{vs::transform([](auto&& i) { return i * i; }),
                               vs::transform([](auto&& i) { return i + i; }),
                               vs::transform([](auto&& i) { return i * i * i; })};

    constexpr auto x =
        tuple{std::array{1, 2, 3}, std::array{2, 3, 4}, std::array{3, 4, 5}};

    REQUIRE((x | tup) ==
            tuple{std::array{1, 4, 9}, std::array{4, 6, 8}, std::array{27, 64, 125}});
}
