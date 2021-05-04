#include "tuple.hpp"

#include "types.hpp"

#include <catch2/catch_test_macros.hpp>

#include <range/v3/all.hpp>

#include <vector>

#include <iostream>

using namespace ccs;

TEST_CASE("concepts")
{
    using T = std::vector<real>;

    REQUIRE(All<T&>);
    REQUIRE(!All<T>);
    REQUIRE(!All<T&&>);

    REQUIRE(!All<container_tuple<T>&>);
    REQUIRE(!All<container_tuple<T>>);

    REQUIRE(All<tuple<T>&>);
    REQUIRE(!All<tuple<T>>);
    REQUIRE(!All<tuple<T>&&>);

    REQUIRE(std::tuple_size_v<tuple<int, int, real>> == 3u);
    REQUIRE(OneTuple<tuple<int>>);
    REQUIRE(!OneTuple<tuple<int, int>>);
    REQUIRE(ThreeTuple<tuple<int, int, real>>);
    REQUIRE(!ThreeTuple<tuple<int>>);

    REQUIRE(OwningTuple<tuple<T>>);
    REQUIRE(!OwningTuple<tuple<T&>>);

    using U = tuple<tuple<T>>;
    REQUIRE(All<U&>);
    REQUIRE(!All<U>);
    REQUIRE(!All<U&&>);

    REQUIRE(All<tuple<U>&>);
    REQUIRE(!All<tuple<U>>);
    REQUIRE(!All<tuple<U>&&>);

    using V = decltype(vs::iota(0, 10));
    REQUIRE(All<V>);
    REQUIRE(All<tuple<V>>);

    using W = std::span<real>;
    REQUIRE(All<W>);
    REQUIRE(All<W&>);
    REQUIRE(All<tuple<W>>);
    REQUIRE(All<tuple<W&>>);
}

TEST_CASE("Default construction")
{
    using T = std::vector<real>;

    {
        T t{};
        REQUIRE(rs::size(t) == 0);
    }

    {
        tuple<T> t{};
        REQUIRE(rs::size(t) == 0);
    }

    {
        tuple<T, T> t{};
        REQUIRE(rs::size(get<0>(t)) == 0);
        REQUIRE(rs::size(get<1>(t)) == 0);
    }

    {
        tuple<tuple<T>, tuple<T, T>> t{};
        REQUIRE(rs::size(get<0>(t)) == 0);
        REQUIRE(rs::size(get<1, 0>(t)) == 0);
        REQUIRE(rs::size(get<1, 1>(t)) == 0);
    }
}

TEST_CASE("construction from subrange")
{
    using T = std::vector<real>;

    // this test highlighted an issue with too broad a definition for tupleLike

    auto x = vs::iota(0, 10) | rs::to<T>();

    auto r = tuple{x | vs::take_exactly(5)};

    REQUIRE(r == vs::iota(0, 5));

    r = 0;
    REQUIRE(x == T{0, 0, 0, 0, 0, 5, 6, 7, 8, 9});
}

TEST_CASE("construction")
{
    auto x = std::vector<real>{0, 1, 2};
    auto y = std::vector<real>{3, 4, 5};
    auto z = std::vector<real>{6, 7, 8};

    auto r = tuple(x);
    REQUIRE(std::same_as<decltype(r), tuple<std::vector<real>&>>);
    REQUIRE(!OwningTuple<decltype(r)>);
    REQUIRE(r[1] == 1);
    r[1] = -1;
    REQUIRE(r[1] == -1);

    REQUIRE(r == x);

    SECTION("owning from size")
    {
        using T = std::vector<real>;
        auto q = tuple<T>{4};
        REQUIRE(q.size() == 4u);

        auto qq = tuple<T, T, T>{4, 5, 6};
        REQUIRE(rs::size(get<0>(qq)) == 4u);
        REQUIRE(rs::size(get<1>(qq)) == 5u);
        REQUIRE(get<2>(qq).size() == 6u);
    }

    SECTION("owning from input range")
    {
        using T = tuple<std::vector<real>>;
        REQUIRE(std::same_as<decltype(get<0>(std::declval<T>())), std::vector<real>&>);
        // try different sizes to flush out memory issues
        for (int i = 10; i < 1024; i += 9) {
            auto xx = tuple<std::vector<real>>{vs::iota(0, i)};
            REQUIRE(xx == vs::iota(0, i));

            xx = vs::iota(0, i / 2);
            REQUIRE(xx == vs::iota(0, i / 2));
        }
    }

    SECTION("owning from non-owning")
    {
        auto r_owning = tuple{std::vector<real>{0, 1, 2}};
        REQUIRE(r_owning[1] == 1);
        r_owning[1] = -1;
        REQUIRE(r_owning[1] == -1);
        REQUIRE(r_owning == x);

        r_owning = r;
        REQUIRE(r_owning == x);
        REQUIRE(r_owning == r);
        auto x0 = x[0];
        // this shouldn't change r/x
        r_owning[0] = 1000;
        REQUIRE(x[0] == x0);
        REQUIRE(r[0] == x0);
        REQUIRE(x0 != 1000);
    }

    SECTION("owning from non-owning 3-tuple")
    {
        auto r_owning =
            tuple{std::vector<real>{}, std::vector<real>{}, std::vector<real>{}};
        auto a = tuple{x, y, z};

        r_owning = a;
        REQUIRE(r_owning == a);
    }

    auto rr = tuple(x, y, z);
    REQUIRE(rs::equal(get<0>(rr), x));
    REQUIRE(rs::equal(get<1>(rr), y));
    REQUIRE(rs::equal(get<2>(rr), z));

    {
        auto xx = x;
        auto yy = y;
        auto zz = z;
        auto r_owning = tuple{MOVE(xx), MOVE(yy), MOVE(zz)};

        REQUIRE(rr == r_owning);
    }

    {
        auto x = tuple{vs::concat(tuple{vs::iota(0, 16)}, vs::iota(16, 21))};
        REQUIRE(x == vs::iota(0, 21));
    }
}

TEST_CASE("Copy and Move Owning Onetuple")
{
    using T = std::vector<real>;

    const auto i = vs::iota(0, 10);
    const auto j = vs::iota(5, 100);

    SECTION("copy assignment")
    {
        tuple<T> x{};
        tuple<T> y{i};
        x = y;
        REQUIRE(x == y);
        REQUIRE(y == i);

        auto z = tuple{j};
        x = z;
        REQUIRE(x == z);
        REQUIRE(z == j);
    }

    SECTION("copy construction")
    {
        tuple<T> y{i};
        tuple<T> x{y};
        REQUIRE(x.size() == y.size());
        REQUIRE(x == i);
        REQUIRE(x == y);

        auto z = tuple{j};
        tuple<T> u{z};
        REQUIRE(u.size() == z.size());
        REQUIRE(u == z);
        REQUIRE(z == j);
    }

    SECTION("move assignment")
    {
        tuple<T> x{};
        tuple<T> y{i};
        x = MOVE(y);
        REQUIRE(x.size() == 10u);
        REQUIRE(x == i);
    }

    SECTION("move construction")
    {
        tuple<T> y{i};
        tuple<T> x{MOVE(y)};
        REQUIRE(x.size() == 10u);
        REQUIRE(x == i);
    }
}

TEST_CASE("Copy and Move Owning TwoTuple")
{
    using T = std::vector<real>;
    using U = tuple<T, T>;

    const auto i = vs::iota(0, 10);
    const auto j = vs::iota(4, 7);

    SECTION("copy assignment")
    {
        U x{};
        U y{i, j};
        x = y;
        REQUIRE(x == y);
    }

    SECTION("copy construction")
    {
        U y{i, j};
        U x{y};
        REQUIRE(x == y);
    }

    SECTION("move assignment")
    {
        U x{};
        U y{i, j};
        x = MOVE(y);
        REQUIRE(x == y);
    }

    SECTION("move construction")
    {
        U y{i, j};
        U x{MOVE(y)};
        REQUIRE(rs::equal(get<0>(x), i));
        REQUIRE(rs::equal(get<1>(x), j));
    }

    SECTION("copy from other")
    {
        U x{};
        auto y = tuple{i, j};
        x = y;
        REQUIRE(x == y);
    }

    SECTION("from generate")
    {
        U x{};
        int k;
        auto g = [&k]() { return ++k; };
        const auto tt = transform(
            [g = g](auto&& s) { return vs::generate_n(g, rs::size(s)) | rs::to<T>(); },
            tuple{i, j});
        REQUIRE(rs::size(get<0>(tt)) == rs::size(i));
        REQUIRE(rs::size(get<1>(tt)) == rs::size(j));
        x = tt;
        REQUIRE(rs::size(get<0>(x)) == rs::size(get<0>(tt)));
        REQUIRE(rs::size(get<1>(x)) == rs::size(get<1>(tt)));
        REQUIRE(x == tt);
    }
}

TEST_CASE("Nested Construction")
{
    using T = std::vector<real>;

    const auto i = vs::iota(0, 10);
    const auto j = vs::iota(1, 5);
    const auto k = vs::iota(7, 50);

    {
        tuple<tuple<T>> x{tuple<T>{i}};
        REQUIRE(x == i);

        auto&& [a] = x;
        REQUIRE(a == i);

        tuple<tuple<T>> y{x};
        REQUIRE(y == x);

        auto&& [b] = y;
        REQUIRE(b == i);
    }

    {
        tuple<tuple<T, T>> x{tuple<T, T>{i, j}};
        REQUIRE(x == tuple{tuple{i, j}});

        auto&& [a, b] = get<0>(x);
        REQUIRE(rs::equal(a, i));
        REQUIRE(rs::equal(b, j));
    }

    {
        tuple<tuple<T>, tuple<T, T>> x{tuple<T>{i}, tuple<T, T>{j, k}};
        REQUIRE(x == tuple{tuple{i}, tuple{j, k}});

        auto&& [a, b] = x;
        REQUIRE(a == i);
        REQUIRE(b == tuple{j, k});
    }

    {
        tuple<tuple<T>, tuple<T, T>> x{tuple{i}, tuple{j, k}};
        REQUIRE(x == tuple{tuple{i}, tuple{j, k}});

        auto&& [a, b] = x;
        REQUIRE(a == i);
        REQUIRE(b == tuple{j, k});
    }
}

TEST_CASE("Conversion OneTuples")
{
    auto x = tuple<std::vector<int>>{std::vector{1, 2, 3}};
    tuple<std::span<int>> y = x;

    REQUIRE(x == y);

    tuple<std::span<int>> yy{};
    yy = x;
    REQUIRE(x == yy);

    tuple<std::span<const int>> z = x;
    REQUIRE(x.size() == z.size());
    REQUIRE(x == z);

    tuple<std::span<const int>> zz = y;
    REQUIRE(x.size() == zz.size());
    REQUIRE(x == zz);

    tuple<std::vector<int>> q = y;
    REQUIRE(q.size() == y.size());
    REQUIRE(q == y);

    auto f = [](tuple<std::span<int>> x) { x = vs::iota(0, (int)x.size()); };
    y = f;
    REQUIRE(x == std::vector{0, 1, 2});
    auto g = [](tuple<std::span<int>> x) { x = vs::iota(1, 1 + (int)x.size()); };
    x = g;
    REQUIRE(y == std::vector{1, 2, 3});
}

TEST_CASE("Conversion ThreeTuples")
{
    using T = std::vector<int>;
    using U = std::span<int>;
    using V = std::span<const int>;

    auto x =
        tuple<T, T, T>{std::vector{1, 2, 3}, std::vector{1}, std::vector{5, 4, 3, 2}};
    tuple<U, U, U> y = x;

    REQUIRE(x == y);

    tuple<V, V, V> z = x;
    REQUIRE(z == x);

    tuple<V, V, V> zz = y;
    REQUIRE(zz == y);

    tuple<T, T, T> q = y;
    REQUIRE(q == y);

    auto f = [](tuple<U, U, U> u) {
        u = tuple{vs::iota(0, 3), vs::iota(2, 3), vs::iota(0, 4)};
    };
    x = f;
    REQUIRE(y == tuple{vs::iota(0, 3), vs::iota(2, 3), vs::iota(0, 4)});
}

TEST_CASE("Conversion Nested ThreeTuples")
{
    using T = std::vector<int>;
    using U = std::span<int>;
    using V = std::span<const int>;

    auto x = tuple<tuple<T>, tuple<T, T, T>>{
        tuple{T{1, 2}},
        tuple{T{1, 2, 3}, T{1}, T{5, 4, 3, 2}},
    };

    using D = list_index<0>;
    using Rx = list_index<1, 0>;
    using Ry = list_index<1, 1>;
    using Rz = list_index<1, 2>;

    tuple<tuple<U>, tuple<U, U, U>> y = x;
    REQUIRE(rs::equal(get<0>(x), get<0>(y)));
    REQUIRE(rs::equal(get<1, 0>(x), get<1, 0>(y)));
    REQUIRE(rs::equal(get<1, 1>(x), get<1, 1>(y)));
    REQUIRE(rs::equal(get<1, 2>(x), get<1, 2>(y)));

    tuple<tuple<V>, tuple<V, V, V>> z = x;
    REQUIRE(rs::equal(get<D>(x), get<D>(z)));
    REQUIRE(rs::equal(get<Rx>(x), get<Rx>(z)));
    REQUIRE(rs::equal(get<Ry>(x), get<Ry>(z)));
    REQUIRE(rs::equal(get<Rz>(x), get<Rz>(z)));

    tuple<tuple<V>, tuple<V, V, V>> zz = y;
    REQUIRE(zz == y);

    tuple<tuple<T>, tuple<T, T, T>> a = x;
    REQUIRE(a == x);

    tuple<tuple<T>, tuple<T, T, T>> b = zz;
    REQUIRE(b == zz);

    auto f = [](tuple<tuple<U>, tuple<U, U, U>> u) {
        u = tuple{tuple{vs::iota(0, 2)},
                  tuple{vs::iota(0, 3), vs::iota(0, 1), vs::iota(0, 4)}};
    };
    x = f;
    REQUIRE(x == tuple{tuple{vs::iota(0, 2)},
                       tuple{vs::iota(0, 3), vs::iota(0, 1), vs::iota(0, 4)}});
}

TEST_CASE("numeric assignment with owning tuple")
{
    const auto zeros = vs::repeat_n(0, 3);
    auto x = tuple{std::vector{1, 2, 3}};
    x = 0;
    REQUIRE(rs::size(x) == 3u);
    REQUIRE(x == zeros);

    auto y = tuple{std::vector{1, 2, 3}, std::vector{4, 5, 6}};
    REQUIRE(y == tuple(vs::iota(1, 4), vs::iota(4, 7)));
    y = 0;
    REQUIRE(y == tuple{zeros, zeros});
}

TEST_CASE("numeric assignment with non-owning tuple")
{
    const auto zeros = vs::repeat_n(0, 3);
    auto v = std::vector{1, 2, 3};

    auto x = tuple{v};
    x = 0;
    REQUIRE(rs::size(x) == 3u);
    REQUIRE(x == zeros);
    REQUIRE(x == v);

    auto u = std::vector{1, 2, 3};
    auto w = std::vector{4, 5, 6};
    auto y = tuple{u, w};
    REQUIRE(y == tuple{vs::iota(1, 4), vs::iota(4, 7)});
    y = 0;
    REQUIRE(y == tuple{zeros, zeros});
    REQUIRE(y == tuple{u, v});
}
