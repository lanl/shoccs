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

#include <iostream>

TEST_CASE("concepts")
{
    using namespace ccs;
    using namespace ccs::field::tuple;
    using T = std::vector<real>;

    REQUIRE(All<T&>);
    REQUIRE(!All<T>);
    REQUIRE(!All<T&&>);

    REQUIRE(!All<ContainerTuple<T>&>);
    REQUIRE(!All<ContainerTuple<T>>);

    REQUIRE(All<Tuple<T>&>);
    REQUIRE(!All<Tuple<T>>);
    REQUIRE(!All<Tuple<T>&&>);

    REQUIRE(std::tuple_size_v<Tuple<int, int, real>> == 3u);
    REQUIRE(traits::OneTuple<Tuple<int>>);
    REQUIRE(!traits::OneTuple<Tuple<int, int>>);
    REQUIRE(traits::ThreeTuple<Tuple<int, int, real>>);
    REQUIRE(!traits::ThreeTuple<Tuple<int>>);

    REQUIRE(traits::OwningTuple<Tuple<T>>);
    REQUIRE(!traits::OwningTuple<Tuple<T&>>);

    using U = Tuple<Tuple<T>>;
    REQUIRE(All<U&>);
    REQUIRE(!All<U>);
    REQUIRE(!All<U&&>);

    REQUIRE(All<Tuple<U>&>);
    REQUIRE(!All<Tuple<U>>);
    REQUIRE(!All<Tuple<U>&&>);

    using V = decltype(vs::iota(0, 10));
    REQUIRE(All<V>);
    REQUIRE(All<Tuple<V>>);

    using W = std::span<real>;
    REQUIRE(All<W>);
    REQUIRE(All<W&>);
    REQUIRE(All<Tuple<W>>);
    REQUIRE(All<Tuple<W&>>);
}

TEST_CASE("construction")
{
    using namespace ccs;
    using namespace ccs::field::tuple;

    auto x = std::vector<real>{0, 1, 2};
    auto y = std::vector<real>{3, 4, 5};
    auto z = std::vector<real>{6, 7, 8};

    auto r = Tuple(x);
    REQUIRE(std::same_as<decltype(r), Tuple<std::vector<real>&>>);
    REQUIRE(!traits::OwningTuple<decltype(r)>);
    REQUIRE(r[1] == 1);
    r[1] = -1;
    REQUIRE(r[1] == -1);

    REQUIRE(rs::equal(r, x));

    SECTION("owning from size")
    {
        using T = std::vector<real>;
        auto q = Tuple<T>{4};
        REQUIRE(q.size() == 4u);

        auto qq = Tuple<T, T, T>{4, 5, 6};
        REQUIRE(rs::size(get<0>(qq)) == 4u);
        REQUIRE(rs::size(get<1>(qq)) == 5u);
        REQUIRE(get<2>(qq).size() == 6u);
    }

    SECTION("owning from input range")
    {
        using T = Tuple<std::vector<real>>;
        REQUIRE(std::same_as<decltype(get<0>(std::declval<T>())), std::vector<real>&>);
        // try different sizes to flush out memory issues
        for (int i = 10; i < 1024; i += 9) {
            auto xx = Tuple<std::vector<real>>{vs::iota(0, i)};
            REQUIRE(rs::equal(vs::iota(0, i), xx));

            xx = vs::iota(0, i / 2);
            REQUIRE(rs::equal(xx, vs::iota(0, i / 2)));
        }
    }

    SECTION("owning from non-owning")
    {
        auto r_owning = Tuple{std::vector<real>{0, 1, 2}};
        REQUIRE(r_owning[1] == 1);
        r_owning[1] = -1;
        REQUIRE(r_owning[1] == -1);
        REQUIRE(rs::equal(r_owning, x));

        r_owning = r;
        REQUIRE(rs::equal(r_owning, x));
        REQUIRE(rs::equal(r_owning, r));
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
            Tuple{std::vector<real>{}, std::vector<real>{}, std::vector<real>{}};
        auto a = Tuple{x, y, z};

        r_owning = a;
        REQUIRE(rs::equal(get<0>(r_owning), x));
        REQUIRE(rs::equal(get<1>(r_owning), y));
        REQUIRE(rs::equal(get<2>(r_owning), z));
    }

    auto rr = Tuple(x, y, z);
    REQUIRE(rs::equal(get<0>(rr), x));
    REQUIRE(rs::equal(get<1>(rr), y));
    REQUIRE(rs::equal(get<2>(rr), z));

    {
        auto xx = x;
        auto yy = y;
        auto zz = z;
        auto r_owning = Tuple{MOVE(xx), MOVE(yy), MOVE(zz)};

        REQUIRE(rs::equal(get<0>(rr), get<0>(r_owning)));
        REQUIRE(rs::equal(get<1>(rr), get<1>(r_owning)));
        REQUIRE(rs::equal(get<2>(rr), get<2>(r_owning)));
    }

    {
        auto x = Tuple{vs::concat(Tuple{vs::iota(0, 16)}, vs::iota(16, 21))};
        REQUIRE(rs::equal(x, vs::iota(0, 21)));
    }
}

TEST_CASE("Copy and Move Owning OneTuple")
{
    using namespace ccs;
    using namespace ccs::field;
    using T = std::vector<real>;

    const auto i = vs::iota(0, 10);
    const auto j = vs::iota(5, 100);

    SECTION("copy assignment")
    {
        Tuple<T> x{};
        Tuple<T> y{i};
        x = y;
        REQUIRE(x.size() == y.size());
        REQUIRE(rs::equal(x, i));
        REQUIRE(rs::equal(y, i));

        auto z = Tuple{j};
        x = z;
        REQUIRE(x.size() == z.size());
        REQUIRE(rs::equal(x, j));
        REQUIRE(rs::equal(z, j));
    }

    SECTION("copy construction")
    {
        Tuple<T> y{i};
        Tuple<T> x{y};
        REQUIRE(x.size() == y.size());
        REQUIRE(rs::equal(x, i));
        REQUIRE(rs::equal(y, i));

        auto z = Tuple{j};
        Tuple<T> u{z};
        REQUIRE(u.size() == z.size());
        REQUIRE(rs::equal(u, j));
        REQUIRE(rs::equal(z, j));
    }

    SECTION("move assignment")
    {
        Tuple<T> x{};
        Tuple<T> y{i};
        x = MOVE(y);
        REQUIRE(x.size() == 10u);
        REQUIRE(rs::equal(x, i));
    }

    SECTION("move construction")
    {
        Tuple<T> y{i};
        Tuple<T> x{MOVE(y)};
        REQUIRE(x.size() == 10u);
        REQUIRE(rs::equal(x, i));
    }
}

TEST_CASE("Copy and Move Owning TwoTuple")
{
    using namespace ccs;
    using namespace ccs::field;
    using T = std::vector<real>;
    using U = Tuple<T, T>;

    const auto i = vs::iota(0, 10);
    const auto j = vs::iota(4, 7);

    SECTION("copy assignment")
    {
        U x{};
        U y{i, j};
        x = y;
        REQUIRE(rs::equal(get<0>(x), i));
        REQUIRE(rs::equal(get<1>(x), j));
    }

    SECTION("copy construction")
    {
        U y{i, j};
        U x{y};
        REQUIRE(rs::equal(get<0>(x), i));
        REQUIRE(rs::equal(get<1>(x), j));
    }

    SECTION("move assignment")
    {
        U x{};
        U y{i, j};
        x = MOVE(y);
        REQUIRE(rs::equal(get<0>(x), i));
        REQUIRE(rs::equal(get<1>(x), j));
    }

    SECTION("move construction")
    {
        U y{i, j};
        U x{MOVE(y)};
        REQUIRE(rs::equal(get<0>(x), i));
        REQUIRE(rs::equal(get<1>(x), j));
    }
}

TEST_CASE("Nested Construction")
{
    using namespace ccs;
    using namespace ccs::field;
    using T = std::vector<real>;

    const auto i = vs::iota(0, 10);
    const auto j = vs::iota(1, 5);
    const auto k = vs::iota(7, 50);

    {
        Tuple<Tuple<T>> x{Tuple<T>{i}};

        auto&& [a] = x;
        REQUIRE(rs::equal(a, i));

        Tuple<Tuple<T>> y{x};
        auto&& [b] = y;
        REQUIRE(rs::equal(b, i));
    }

    {
        Tuple<Tuple<T, T>> x{Tuple<T, T>{i, j}};

        auto&& [a, b] = get<0>(x);
        REQUIRE(rs::equal(a, i));
        REQUIRE(rs::equal(b, j));
    }

    {
        Tuple<Tuple<T>, Tuple<T, T>> x{Tuple<T>{i}, Tuple<T, T>{j, k}};

        auto&& [a, b] = x;
        REQUIRE(rs::equal(a, i));
        REQUIRE(rs::equal(get<0>(b), j));
        REQUIRE(rs::equal(get<1>(b), k));
    }

    {
        Tuple<Tuple<T>, Tuple<T, T>> x{Tuple{i}, Tuple{j, k}};

        auto&& [a, b] = x;
        REQUIRE(rs::equal(a, i));
        REQUIRE(rs::equal(get<0>(b), j));
        REQUIRE(rs::equal(get<1>(b), k));
    }
}

TEST_CASE("Conversion OneTuples")
{
    using namespace ccs;
    using namespace ccs::field;

    auto x = Tuple<std::vector<int>>{std::vector{1, 2, 3}};
    Tuple<std::span<int>> y = x;

    REQUIRE(x.size() == y.size());
    REQUIRE(rs::equal(x, y));

    Tuple<std::span<int>> yy{};
    yy = x;
    REQUIRE(x.size() == yy.size());
    REQUIRE(rs::equal(x, yy));

    Tuple<std::span<const int>> z = x;
    REQUIRE(x.size() == z.size());
    REQUIRE(rs::equal(x, z));

    Tuple<std::span<const int>> zz = y;
    REQUIRE(x.size() == zz.size());
    REQUIRE(rs::equal(x, zz));

    Tuple<std::vector<int>> q = y;
    REQUIRE(q.size() == y.size());
    REQUIRE(rs::equal(q, y));
}

TEST_CASE("Conversion ThreeTuples")
{
    using namespace ccs;
    using namespace ccs::field;
    using T = std::vector<int>;
    using U = std::span<int>;
    using V = std::span<const int>;

    auto x =
        Tuple<T, T, T>{std::vector{1, 2, 3}, std::vector{1}, std::vector{5, 4, 3, 2}};
    Tuple<U, U, U> y = x;

    auto&& [a, b, c] = x;

    REQUIRE(rs::equal(a, get<0>(y)));
    REQUIRE(rs::equal(b, get<1>(y)));
    REQUIRE(rs::equal(c, get<2>(y)));

    Tuple<V, V, V> z = x;
    REQUIRE(rs::equal(a, get<0>(z)));
    REQUIRE(rs::equal(b, get<1>(z)));
    REQUIRE(rs::equal(c, get<2>(z)));

    Tuple<V, V, V> zz = y;
    REQUIRE(rs::equal(a, get<0>(zz)));
    REQUIRE(rs::equal(b, get<1>(zz)));
    REQUIRE(rs::equal(c, get<2>(zz)));

    Tuple<T, T, T> q = y;
}

TEST_CASE("Conversion Nested ThreeTuples")
{
    using namespace ccs;
    using namespace ccs::field;
    using T = std::vector<int>;
    using U = std::span<int>;
    using V = std::span<const int>;

    auto x = Tuple<Tuple<T>, Tuple<T, T, T>>{
        Tuple{T{1, 2}},
        Tuple{T{1, 2, 3}, T{1}, T{5, 4, 3, 2}},
    };

    Tuple<Tuple<U>, Tuple<U, U, U>> y = x;
    REQUIRE(rs::equal(get<0>(x), get<0>(y)));
    REQUIRE(rs::equal(get<0, 1>(x), get<0, 1>(y)));
    REQUIRE(rs::equal(get<1, 1>(x), get<1, 1>(y)));
    REQUIRE(rs::equal(get<2, 1>(x), get<2, 1>(y)));

    Tuple<Tuple<V>, Tuple<V, V, V>> z = x;
    REQUIRE(rs::equal(get<0>(x), get<0>(z)));
    REQUIRE(rs::equal(get<0, 1>(x), get<0, 1>(z)));
    REQUIRE(rs::equal(get<1, 1>(x), get<1, 1>(z)));
    REQUIRE(rs::equal(get<2, 1>(x), get<2, 1>(z)));

    Tuple<Tuple<V>, Tuple<V, V, V>> zz = y;
    REQUIRE(rs::equal(get<0>(x), get<0>(zz)));
    REQUIRE(rs::equal(get<0, 1>(x), get<0, 1>(zz)));
    REQUIRE(rs::equal(get<1, 1>(x), get<1, 1>(zz)));
    REQUIRE(rs::equal(get<2, 1>(x), get<2, 1>(zz)));

    Tuple<Tuple<T>, Tuple<T, T, T>> a = x;
    REQUIRE(rs::equal(get<0>(x), get<0>(a)));
    REQUIRE(rs::equal(get<0, 1>(x), get<0, 1>(a)));
    REQUIRE(rs::equal(get<1, 1>(x), get<1, 1>(a)));
    REQUIRE(rs::equal(get<2, 1>(x), get<2, 1>(a)));

    Tuple<Tuple<T>, Tuple<T, T, T>> b = zz;
    REQUIRE(rs::equal(get<0>(x), get<0>(b)));
    REQUIRE(rs::equal(get<0, 1>(x), get<0, 1>(b)));
    REQUIRE(rs::equal(get<1, 1>(x), get<1, 1>(b)));
    REQUIRE(rs::equal(get<2, 1>(x), get<2, 1>(b)));
}

TEST_CASE("numeric assignment with owning tuple")
{
    using namespace ccs;
    using namespace ccs::field;

    const auto zeros = vs::repeat_n(0, 3);
    auto x = Tuple{std::vector{1, 2, 3}};
    x = 0;
    REQUIRE(rs::size(x) == 3u);
    REQUIRE(rs::equal(x, zeros));

    auto y = Tuple{std::vector{1, 2, 3}, std::vector{4, 5, 6}};
    REQUIRE(rs::equal(get<1>(y), std::vector{4, 5, 6}));
    y = 0;
    REQUIRE(rs::equal(get<0>(y), zeros));
    REQUIRE(rs::equal(get<1>(y), zeros));
}

TEST_CASE("numeric assignment with non-owning tuple")
{
    using namespace ccs;
    using namespace ccs::field;

    const auto zeros = vs::repeat_n(0, 3);
    auto v = std::vector{1, 2, 3};

    auto x = Tuple{v};
    x = 0;
    REQUIRE(rs::size(x) == 3u);
    REQUIRE(rs::equal(x, zeros));
    REQUIRE(rs::equal(x, v));

    auto u = std::vector{1, 2, 3};
    auto w = std::vector{4, 5, 6};
    auto y = Tuple{u, w};
    REQUIRE(rs::equal(get<1>(y), w));
    y = 0;
    REQUIRE(rs::equal(get<0>(y), zeros));
    REQUIRE(rs::equal(get<1>(y), zeros));
    REQUIRE(rs::equal(get<1>(y), w));
}
