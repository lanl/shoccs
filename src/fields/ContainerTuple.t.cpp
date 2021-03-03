#include "ContainerTuple.hpp"
#include "ViewTuple.hpp"

#include <catch2/catch_test_macros.hpp>

#include <range/v3/algorithm/equal.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/repeat_n.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/zip_with.hpp>

#include <vector>

TEST_CASE("Structured Binding")
{
    using namespace ccs;
    using namespace ccs::field::tuple;

    using T = std::vector<real>;

    auto x = ContainerTuple<T, T>{std::vector<real>{0, 1, 2}, std::vector<real>{3, 4, 5}};
    static_assert(std::tuple_size_v<std::remove_cvref_t<decltype(x)>> == 2u);

    {
        // get references to the ranges, changes are visible
        auto&& [a, b] = x;
        for (auto&& i : a) i *= 3;
        b[0] = 1;
        b[1] = 2;
        b[2] = 3;
    }

    auto&& [a, b] = x;
    REQUIRE(rs::equal(a, std::vector<real>{0, 3, 6}));
    REQUIRE(rs::equal(b, std::vector<real>{1, 2, 3}));

    {
        // copy the ranges, changes are not visible
        auto [a, b] = x;
        for (auto&& i : a) i *= 3;
        b[0] = 1;
        b[1] = 2;
        b[2] = 3;
    }

    auto&& [c, d] = x;
    REQUIRE(rs::equal(c, std::vector<real>{0, 3, 6}));
    REQUIRE(rs::equal(d, std::vector<real>{1, 2, 3}));
}

TEST_CASE("Assignment")
{
    using namespace ccs;
    using namespace ccs::field::tuple;

    using T = std::vector<real>;

    auto x = ContainerTuple<T, T>{std::vector<real>{0, 1, 2}, std::vector<real>{3, 4, 5}};
    x = 1;

    REQUIRE(rs::equal(get<0>(x), std::vector<real>{1, 1, 1}));
}

TEST_CASE("Construction from Ranges")
{
    using namespace ccs;
    using namespace ccs::field::tuple;

    using T = std::vector<real>;
    auto x = ContainerTuple<T>{vs::iota(0, 10)};
    {
        auto&& [v] = x;
        REQUIRE(rs::equal(v, vs::iota(0, 10)));
    }

    auto y = ContainerTuple<T, T>{vs::iota(0, 10), vs::iota(1, 6)};
    {
        REQUIRE(rs::equal(get<0>(y), vs::iota(0, 10)));
        REQUIRE(rs::equal(get<1>(y), vs::iota(1, 6)));
    }
}

TEST_CASE("Construction from OneViewTuples")
{
    using namespace ccs;
    using namespace ccs::field::tuple;

    using T = std::vector<real>;
    const auto i = vs::iota(0, 100);

    {
        auto v = ViewTuple{i};
        ContainerTuple<T> x{};
        x = v;
        REQUIRE(rs::equal(get<0>(x), i));
    }

    {
        auto v = ViewTuple{i};
        ContainerTuple<T> x{v};
        REQUIRE(rs::equal(get<0>(x), i));
    }

    {
        auto v = ViewTuple{i};
        ContainerTuple<T> x{};
        x = MOVE(v);
        REQUIRE(rs::equal(get<0>(x), i));
    }

    {
        auto v = ViewTuple{i};
        ContainerTuple<T> x{MOVE(v)};
        REQUIRE(rs::equal(get<0>(x), i));
    }
}

TEST_CASE("Construction from TwoViewTuples")
{
    using namespace ccs;
    using namespace ccs::field::tuple;

    using T = std::vector<real>;
    using U = ContainerTuple<T, T>;
    const auto i = vs::iota(0, 100);
    const auto j = vs::iota(-1, 1);

    {
        auto v = ViewTuple{i, j};
        U x{};
        x = v;
        REQUIRE(rs::equal(get<0>(x), i));
        REQUIRE(rs::equal(get<1>(x), j));
    }

    {
        auto v = ViewTuple{i, j};
        U x{v};
        REQUIRE(rs::equal(get<0>(x), i));
        REQUIRE(rs::equal(get<1>(x), j));
    }

    {
        auto v = ViewTuple{i, j};
        U x{};
        x = MOVE(v);
        REQUIRE(rs::equal(get<0>(x), i));
        REQUIRE(rs::equal(get<1>(x), j));
    }

    {
        auto v = ViewTuple{i, j};
        U x{MOVE(v)};
        REQUIRE(rs::equal(get<0>(x), i));
        REQUIRE(rs::equal(get<1>(x), j));
    }
}

TEST_CASE("Copy")
{
    using namespace ccs;
    using namespace ccs::field::tuple;

    using T = std::vector<real>;

    SECTION("copy assignment OneTuple")
    {
        auto x = ContainerTuple<T>{};
        const auto y = ContainerTuple<T>{vs::iota(0, 10)};

        x = y;
        REQUIRE(rs::equal(get<0>(x), get<0>(y)));
    }
    SECTION("copy construction OneTuple")
    {
        const auto y = ContainerTuple<T>{vs::iota(0, 10)};
        ContainerTuple<T> x{y};

        REQUIRE(rs::equal(get<0>(x), get<0>(y)));
    }

    SECTION("copy assignment TwoTuple")
    {
        auto x = ContainerTuple<T, T>{};
        const auto y = ContainerTuple<T, T>{vs::iota(0, 10), vs::iota(1, 4)};

        x = y;
        REQUIRE(rs::equal(get<0>(x), get<0>(y)));
        REQUIRE(rs::equal(get<1>(x), get<1>(y)));
    }

    SECTION("copy construction TwoTuple")
    {
        const auto y = ContainerTuple<T, T>{vs::iota(0, 10), vs::iota(1, 4)};
        ContainerTuple<T, T> x{y};

        REQUIRE(rs::equal(get<0>(x), get<0>(y)));
        REQUIRE(rs::equal(get<1>(x), get<1>(y)));
    }

    SECTION("copy assignment for OtherContainerTuple")
    {
        auto x = ContainerTuple<T, T>{};
        const auto y = ContainerTuple<std::vector<int>, std::vector<int>>{vs::iota(0, 10),
                                                                          vs::iota(1, 4)};

        x = y;
        REQUIRE(rs::equal(get<0>(x), get<0>(y)));
        REQUIRE(rs::equal(get<1>(x), get<1>(y)));
    }
}

TEST_CASE("Move")
{
    using namespace ccs;
    using namespace ccs::field::tuple;

    using T = std::vector<real>;

    const auto i = vs::iota(0, 10);
    const auto j = vs::iota(1, 4);

    SECTION("move assignment OneTuple")
    {
        auto x = ContainerTuple<T>{};
        const auto y = ContainerTuple<T>{i};

        x = MOVE(y);
        REQUIRE(rs::equal(get<0>(x), i));
    }
    SECTION("move construction OneTuple")
    {
        auto y = ContainerTuple<T>{i};
        ContainerTuple<T> x{MOVE(y)};

        REQUIRE(rs::equal(get<0>(x), i));
    }

    SECTION("move assignment TwoTuple")
    {
        auto x = ContainerTuple<T, T>{};
        auto y = ContainerTuple<T, T>{i, j};

        x = MOVE(y);
        REQUIRE(rs::equal(get<0>(x), i));
        REQUIRE(rs::equal(get<1>(x), j));
    }

    SECTION("move construction TwoTuple")
    {
        auto y = ContainerTuple<T, T>{i, j};
        ContainerTuple<T, T> x{MOVE(y)};

        REQUIRE(rs::equal(get<0>(x), i));
        REQUIRE(rs::equal(get<1>(x), j));
    }

    SECTION("move assignment for OtherContainerTuple")
    {
        auto x = ContainerTuple<T, T>{};
        auto y = ContainerTuple<std::vector<int>, std::vector<int>>{i, j};

        x = MOVE(y);
        REQUIRE(rs::equal(get<0>(x), i));
        REQUIRE(rs::equal(get<1>(x), j));
    }
}

TEST_CASE("Nested")
{
    using namespace ccs;
    using namespace field::tuple;

    using T = std::vector<real>;

    {
        ContainerTuple<ContainerTuple<T, T>> x{
            ContainerTuple<T, T>{vs::iota(0, 10), vs::iota(4, 7)}};

        auto&& [c] = x;

        auto&& [a, b] = c;

        REQUIRE(rs::equal(a, vs::iota(0, 10)));
        REQUIRE(rs::equal(b, vs::iota(4, 7)));
    }

    {
        ContainerTuple<ContainerTuple<T, T>, ContainerTuple<T>> x{
            ContainerTuple<T, T>{vs::iota(0, 10), vs::iota(4, 7)},
            ContainerTuple<T>{T{-1, -2, -3}}};

        auto&& [u, v] = x;

        auto&& [a, b] = u;
        REQUIRE(rs::equal(a, vs::iota(0, 10)));
        REQUIRE(rs::equal(b, vs::iota(4, 7)));

        auto&& [c] = v;
        REQUIRE(rs::equal(c, T{-1, -2, -3}));
    }
}