#include "Tuple.hpp"

#include "types.hpp"

#include <catch2/catch_test_macros.hpp>

#include <range/v3/algorithm/equal.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/zip_with.hpp>

#include <vector>

TEST_CASE("container math concepts")
{
    using namespace ccs;
    using namespace ccs::field::tuple;

    constexpr auto f = []<typename T, bool B = true>()
    {
        if constexpr (B) {
            static_assert(requires(Tuple<T> x) { x = 1; });
            static_assert(requires(Tuple<T> x) { x += 1; });
        } else {
            static_assert(!requires(Tuple<T> x) { x = 1; });
            static_assert(!requires(Tuple<T> x) { x += 1; });
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
    using namespace ccs;
    using namespace ccs::field::tuple;
    using T = std::vector<real>;

    constexpr auto plus = [](auto&& a, auto&& b) {
        return vs::zip_with(std::plus{}, FWD(a), FWD(b));
    };

    auto x = Tuple<T>{vs::iota(0, 10)};
    x += 5;
    REQUIRE(x == vs::iota(5, 15));

    Tuple<T> y = Tuple{vs::iota(0, 10)};
    y += 5;
    REQUIRE(x == y);

    auto xx = Tuple<T, T>{vs::iota(-5, 3), vs::iota(-10, 1)};
    auto yy = Tuple{vs::iota(-5, 3), vs::iota(-10, 1)};
    xx += 100;
    REQUIRE(xx == Tuple{vs::iota(95, 103), vs::iota(90, 101)});

    xx += yy;
    REQUIRE(xx == Tuple{plus(vs::iota(95, 103), vs::iota(-5, 3)),
                        plus(vs::iota(90, 101), vs::iota(-10, 1))});
    // nested;
    const auto i = vs::iota(0, 10);
    const auto j = vs::iota(-1, 10);
    const auto k = vs::iota(-2, 20);
    Tuple<Tuple<T>, Tuple<T, T>> a{Tuple{i}, Tuple{j, k}};
    a += 1;
    REQUIRE(a == Tuple{Tuple{vs::iota(1, 11)}, Tuple{vs::iota(0, 11), vs::iota(-1, 21)}});

    Tuple<Tuple<T>, Tuple<T, T>> b{Tuple{i}, Tuple{j, k}};
    a += b;
    REQUIRE(a == Tuple{Tuple{plus(vs::iota(1, 11), i)},
                       Tuple{plus(vs::iota(0, 11), j), plus(vs::iota(-1, 21), k)}});
}

TEST_CASE("conversions with math")
{
    using namespace ccs;
    using namespace ccs::field::tuple;
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
        Tuple<T> x{i};
        auto q = x + 1;
        auto r = q + x;
        Tuple<T> y{r};
        REQUIRE(y == plus(vs::iota(1, 11), i));

        Tuple<T> z = r;
        REQUIRE(z == r);

        Tuple<T> a{};
        a = r;
        REQUIRE(a == r);

        Tuple<T> b{rs::size(a)};
        Tuple<U> c = b;
        c = r;
        REQUIRE(c == r);
    }

    SECTION("two tuple")
    {
        Tuple<T, T> x{i, j};
        auto q = x + 1;
        auto r = q + x;
        Tuple<T, T> y{r};
        REQUIRE(y == Tuple{plus(vs::iota(1, 11), i), plus(vs::iota(6, 8), j)});

        Tuple<T, T> z = r;
        REQUIRE(z == r);

        Tuple<T, T> a{};
        a = r;
        REQUIRE(a == r);

        Tuple<T, T> b{x};
        Tuple<U, U> c = b;
        c = r;
        REQUIRE(c == r);
    }

    SECTION("Nested")
    {
        using namespace traits;
        using V = Tuple<Tuple<T>, Tuple<T, T>>;
        V x{Tuple{i}, Tuple{j, k}};
        auto q = x + 1;
        auto r = q + x;

        V y{r};
        REQUIRE(y == Tuple{Tuple{plus(vs::iota(1, 11), i)},
                           Tuple{plus(vs::iota(6, 8), j), plus(vs::iota(51, 101), k)}});

        V z = r;
        REQUIRE(z == r);

        V a{};
        a = r;
        REQUIRE(a == r);

        V b{x};
        Tuple<Tuple<U>, Tuple<U, U>> c = b;
        c = r;
        REQUIRE(c == r);
    }
}

TEST_CASE("view math with numeric")
{
    using namespace ccs;
    using namespace ccs::field::tuple;

    constexpr auto plus = [](auto&& a, auto&& b) {
        return vs::zip_with(std::plus{}, FWD(a), FWD(b));
    };

    using T = std::vector<real>;

    for (int i = 100; i < 1000; i += 10) {
        {
            auto x = Tuple<T>{vs::iota(-i, i)};
            auto z = 5 + x + 5;

            REQUIRE(z == vs::iota(10 - i, 10 + i));
            REQUIRE(z == vs::iota(10 - i, 10 + i));
        }

        {
            auto vx = vs::iota(-i, 0);
            auto vy = vs::iota(0, i);
            auto vz = vs::iota(-1, 2 * i);
            auto xx = Tuple<T, T, T>{vx, vy, vz};
            auto zz = (xx + 5) + 1;

            auto o = vs::repeat(6);

            REQUIRE(zz == Tuple{plus(vx, o), plus(vy, o), plus(vz, o)});
        }

        {
            auto x = vs::iota(0, i);
            auto y = vs::iota(i, 2 * i);
            auto z = vs::iota(2 * i, 3 * i);
            auto xyz = Tuple<T, T, T>(x, y, z);

            auto s = xyz + Tuple{x, y, z};
            REQUIRE(s == Tuple{plus(x, x), plus(y, y), plus(z, z)});
        }

        {
            auto x = vs::iota(0, i);
            auto y = vs::iota(i, 2 * i);
            auto z = vs::iota(2 * i, 3 * i);
            auto xyz = Tuple<Tuple<T>, Tuple<T, T>>{Tuple{x}, Tuple{y, z}};

            auto s = xyz + Tuple{Tuple{x}, Tuple{y, z}};
            REQUIRE(s == Tuple{Tuple{plus(x, x)}, Tuple{plus(y, y), plus(z, z)}});
        }
    }
}

TEST_CASE("view math with tuples")
{
    using namespace ccs;
    using namespace ccs::field::tuple;

    using T = std::vector<real>;

    auto x = Tuple<T>{vs::iota(0, 10)};
    auto y = Tuple{vs::iota(10, 20)};
    auto z = x + y;

    REQUIRE(z == vs::zip_with(std::plus{}, vs::iota(0, 10), vs::iota(10, 20)));

    auto g = z + x + x + y;
    auto h = z + z + x;

    REQUIRE(g.size() == h.size());
    REQUIRE(g == h);

    {
        auto xx = Tuple<T>{x + y};
        REQUIRE(z == xx);

        auto yy = Tuple<T>{};
        yy = x + y;
        REQUIRE(yy == xx);
    }
}
