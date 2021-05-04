#include "tuple.hpp"

#include "types.hpp"

#include <catch2/catch_test_macros.hpp>

#include <range/v3/algorithm/equal.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/zip_with.hpp>

#include <vector>

using namespace ccs;

TEST_CASE("container math concepts")
{
    constexpr auto f = []<typename T, bool B = true>()
    {
        if constexpr (B) {
            static_assert(requires(tuple<T> x) { x = 1; });
            static_assert(requires(tuple<T> x) { x += 1; });
        } else {
            static_assert(!requires(tuple<T> x) { x = 1; });
            static_assert(!requires(tuple<T> x) { x += 1; });
        }
    };
    f.template operator()<std::vector<real>>();
    f.template operator()<std::vector<real>&>();
    f.template operator()<const std::vector<real>&, false>();
    f.template operator()<std::span<real>>();
    f.template operator()<std::span<const real>, false>();
}

TEST_CASE("container math")
{
    using T = std::vector<real>;

    constexpr auto plus = [](auto&& a, auto&& b) {
        return vs::zip_with(std::plus{}, FWD(a), FWD(b));
    };

    auto x = tuple<T>{vs::iota(0, 10)};
    x += 5;
    REQUIRE(x == vs::iota(5, 15));

    tuple<T> y = tuple{vs::iota(0, 10)};
    y += 5;
    REQUIRE(x == y);

    auto xx = tuple<T, T>{vs::iota(-5, 3), vs::iota(-10, 1)};
    auto yy = tuple{vs::iota(-5, 3), vs::iota(-10, 1)};
    xx += 100;
    REQUIRE(xx == tuple{vs::iota(95, 103), vs::iota(90, 101)});

    xx += yy;
    REQUIRE(xx == tuple{plus(vs::iota(95, 103), vs::iota(-5, 3)),
                        plus(vs::iota(90, 101), vs::iota(-10, 1))});
    // nested;
    const auto i = vs::iota(0, 10);
    const auto j = vs::iota(-1, 10);
    const auto k = vs::iota(-2, 20);
    tuple<tuple<T>, tuple<T, T>> a{tuple{i}, tuple{j, k}};
    a += 1;
    REQUIRE(a == tuple{tuple{vs::iota(1, 11)}, tuple{vs::iota(0, 11), vs::iota(-1, 21)}});

    tuple<tuple<T>, tuple<T, T>> b{tuple{i}, tuple{j, k}};
    a += b;
    REQUIRE(a == tuple{tuple{plus(vs::iota(1, 11), i)},
                       tuple{plus(vs::iota(0, 11), j), plus(vs::iota(-1, 21), k)}});
}

TEST_CASE("conversions with math")
{
    using T = std::vector<real>;
    using U = std::span<real>;

    const auto i = vs::iota(0, 10);
    const auto j = vs::iota(5, 7);
    const auto k = vs::iota(50, 100);

    constexpr auto plus = [](auto&& a, auto&& b) {
        return vs::zip_with(std::plus{}, FWD(a), FWD(b));
    };

    SECTION("one tuple")
    {
        tuple<T> x{i};
        auto q = x + 1;
        auto r = q + x;
        tuple<T> y{r};
        REQUIRE(y == plus(vs::iota(1, 11), i));

        tuple<T> z = r;
        REQUIRE(z == r);

        tuple<T> a{};
        a = r;
        REQUIRE(a == r);

        tuple<T> b{rs::size(a)};
        tuple<U> c = b;
        c = r;
        REQUIRE(c == r);
    }

    SECTION("two tuple")
    {
        tuple<T, T> x{i, j};
        auto q = x + 1;
        auto r = q + x;
        tuple<T, T> y{r};
        REQUIRE(y == tuple{plus(vs::iota(1, 11), i), plus(vs::iota(6, 8), j)});

        tuple<T, T> z = r;
        REQUIRE(z == r);

        tuple<T, T> a{};
        a = r;
        REQUIRE(a == r);

        tuple<T, T> b{x};
        tuple<U, U> c = b;
        c = r;
        REQUIRE(c == r);
    }

    SECTION("Nested")
    {
        using V = tuple<tuple<T>, tuple<T, T>>;
        V x{tuple{i}, tuple{j, k}};
        auto q = x + 1;
        auto r = q + x;

        V y{r};
        REQUIRE(y == tuple{tuple{plus(vs::iota(1, 11), i)},
                           tuple{plus(vs::iota(6, 8), j), plus(vs::iota(51, 101), k)}});

        V z = r;
        REQUIRE(z == r);

        V a{};
        a = r;
        REQUIRE(a == r);

        V b{x};
        tuple<tuple<U>, tuple<U, U>> c = b;
        c = r;
        REQUIRE(c == r);
    }
}

TEST_CASE("view math with numeric")
{
    constexpr auto plus = [](auto&& a, auto&& b) {
        return vs::zip_with(std::plus{}, FWD(a), FWD(b));
    };

    using T = std::vector<real>;

    for (int i = 100; i < 1000; i += 10) {
        {
            auto x = tuple<T>{vs::iota(-i, i)};
            auto z = 5 + x + 5;

            REQUIRE(z == vs::iota(10 - i, 10 + i));
            REQUIRE(z == vs::iota(10 - i, 10 + i));
        }

        {
            auto vx = vs::iota(-i, 0);
            auto vy = vs::iota(0, i);
            auto vz = vs::iota(-1, 2 * i);
            auto xx = tuple<T, T, T>{vx, vy, vz};
            auto zz = (xx + 5) + 1;

            auto o = vs::repeat(6);

            REQUIRE(zz == tuple{plus(vx, o), plus(vy, o), plus(vz, o)});
        }

        {
            auto x = vs::iota(0, i);
            auto y = vs::iota(i, 2 * i);
            auto z = vs::iota(2 * i, 3 * i);
            auto xyz = tuple<T, T, T>(x, y, z);

            auto s = xyz + tuple{x, y, z};
            REQUIRE(s == tuple{plus(x, x), plus(y, y), plus(z, z)});
        }

        {
            auto x = vs::iota(0, i);
            auto y = vs::iota(i, 2 * i);
            auto z = vs::iota(2 * i, 3 * i);
            auto xyz = tuple<tuple<T>, tuple<T, T>>{tuple{x}, tuple{y, z}};

            auto s = xyz + tuple{tuple{x}, tuple{y, z}};
            REQUIRE(s == tuple{tuple{plus(x, x)}, tuple{plus(y, y), plus(z, z)}});
        }
    }
}

TEST_CASE("view math with tuples")
{
    using T = std::vector<real>;

    auto x = tuple<T>{vs::iota(0, 10)};
    auto y = tuple{vs::iota(10, 20)};
    auto z = x + y;

    REQUIRE(z == vs::zip_with(std::plus{}, vs::iota(0, 10), vs::iota(10, 20)));

    auto g = z + x + x + y;
    auto h = z + z + x;

    REQUIRE(g.size() == h.size());
    REQUIRE(g == h);

    {
        auto xx = tuple<T>{x + y};
        REQUIRE(z == xx);

        auto yy = tuple<T>{};
        yy = x + y;
        REQUIRE(yy == xx);
    }
}
