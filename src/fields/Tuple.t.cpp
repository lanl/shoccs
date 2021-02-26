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

    REQUIRE(!All<container_tuple<T>&>);
    REQUIRE(!All<container_tuple<T>>);

    REQUIRE(All<Tuple<T>&>);
    REQUIRE(!All<Tuple<T>>);
    REQUIRE(!All<Tuple<T>&&>);

    REQUIRE(std::tuple_size_v<Tuple<int, int, real>> == 3u);
    REQUIRE(traits::OneTuple<Tuple<int>>);
    REQUIRE(!traits::OneTuple<Tuple<int, int>>);
    REQUIRE(traits::ThreeTuple<Tuple<int, int, real>>);
    REQUIRE(!traits::ThreeTuple<Tuple<int>>);

    REQUIRE(traits::Owning_Tuple<Tuple<T>>);
    REQUIRE(!traits::Owning_Tuple<Tuple<T&>>);

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
    REQUIRE(!traits::Owning_Tuple<decltype(r)>);
    REQUIRE(r[1] == 1);
    r[1] = -1;
    REQUIRE(r[1] == -1);

    REQUIRE(rs::equal(view<0>(r), x));

    SECTION("owning from size")
    {
        using T = std::vector<real>;
        auto q = Tuple<T>{4};
        REQUIRE(q.size() == 4u);

        Tuple<T> t = Tuple{4};
        REQUIRE(t.size() == 4u);

        auto qq = Tuple<T, T, T>{4, 5, 6};
        REQUIRE(rs::size(view<0>(qq)) == 4u);
        REQUIRE(rs::size(view<1>(qq)) == 5u);
        REQUIRE(qq.get<2>().size() == 6u);

        Tuple<T, T, T> tt = Tuple{7, 8, 9};
        REQUIRE(rs::size(view<0>(tt)) == 7u);
        REQUIRE(rs::size(view<1>(tt)) == 8u);
        REQUIRE(tt.get<2>().size() == 9u);
    }

    SECTION("Nested owning from size")
    {
        using T = std::vector<real>;
        using U = Tuple<Tuple<T>>;

        auto u = U{10};
        REQUIRE(u.size() == 10u);

        using V = Tuple<Tuple<T, T, T>>;
        V v{tag{}, Tuple{11, 12, 13}};

        const auto& g = v.get<0>();
        REQUIRE(rs::size(view<0>(g)) == 11u);
        REQUIRE(rs::size(view<1>(g)) == 12u);
        REQUIRE(g.get<2>().size() == 13u);
    }

    SECTION("owning from input range")
    {
        using T = Tuple<std::vector<real>>;
        REQUIRE(std::same_as<decltype(std::declval<T>().template get<0>()),
                             std::vector<real>&>);
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
        REQUIRE(rs::equal(view<0>(r_owning), x));

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
        REQUIRE(rs::equal(view<0>(r_owning), x));
        REQUIRE(rs::equal(view<1>(r_owning), y));
        REQUIRE(rs::equal(view<2>(r_owning), z));
    }

    auto rr = Tuple(x, y, z);
    REQUIRE(rr.N == 3);
    REQUIRE(rs::equal(view<0>(rr), x));
    REQUIRE(rs::equal(view<1>(rr), y));
    REQUIRE(rs::equal(view<2>(rr), z));

    {
        auto xx = x;
        auto yy = y;
        auto zz = z;
        auto r_owning = Tuple{MOVE(xx), MOVE(yy), MOVE(zz)};
        REQUIRE(r_owning.N == 3);
        REQUIRE(rs::equal(view<0>(rr), view<0>(r_owning)));
        REQUIRE(rs::equal(view<1>(rr), view<1>(r_owning)));
        REQUIRE(rs::equal(view<2>(rr), view<2>(r_owning)));
    }

    {
        auto x = Tuple{vs::concat(Tuple{vs::iota(0, 16)}, vs::iota(16, 21))};
        REQUIRE(rs::equal(x, vs::iota(0, 21)));
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

    Tuple<std::span<const int>> z = x;
    REQUIRE(x.size() == z.size());
    REQUIRE(rs::equal(x, z));

    Tuple<std::span<const int>> zz = y;
    REQUIRE(x.size() == zz.size());
    REQUIRE(rs::equal(x, zz));
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

    REQUIRE(rs::equal(view<0>(x), view<0>(y)));
    REQUIRE(rs::equal(view<1>(x), view<1>(y)));
    REQUIRE(rs::equal(view<2>(x), view<2>(y)));

    Tuple<V, V, V> z = x;
    REQUIRE(rs::equal(view<0>(x), view<0>(z)));
    REQUIRE(rs::equal(view<1>(x), view<1>(z)));
    REQUIRE(rs::equal(view<2>(x), view<2>(z)));

    Tuple<V, V, V> zz = y;
    REQUIRE(rs::equal(view<0>(x), view<0>(zz)));
    REQUIRE(rs::equal(view<1>(x), view<1>(zz)));
    REQUIRE(rs::equal(view<2>(x), view<2>(zz)));
}

TEST_CASE("Conversion Nested ThreeTuples")
{
    using namespace ccs;
    using namespace ccs::field;
    using T = std::vector<int>;
    using U = std::span<int>;
    using V = std::span<const int>;

    auto x = Tuple<Tuple<T>, Tuple<T, T, T>>{
        Tuple{std::vector{1, 2}},
        Tuple{std::vector{1, 2, 3}, std::vector{1}, std::vector{5, 4, 3, 2}},
    };

    Tuple<Tuple<U>, Tuple<U, U, U>> y = x;
    REQUIRE(rs::equal(x.get<0>(), y.get<0>()));
    REQUIRE(rs::equal(view<0>(x.get<1>()), view<0>(y.get<1>())));
    REQUIRE(rs::equal(view<1>(x.get<1>()), view<1>(y.get<1>())));
    REQUIRE(rs::equal(view<2>(x.get<1>()), view<2>(y.get<1>())));

    Tuple<Tuple<V>, Tuple<V, V, V>> z = x;
    REQUIRE(rs::equal(x.get<0>(), z.get<0>()));
    REQUIRE(rs::equal(view<0>(x.get<1>()), view<0>(z.get<1>())));
    REQUIRE(rs::equal(view<1>(x.get<1>()), view<1>(z.get<1>())));
    REQUIRE(rs::equal(view<2>(x.get<1>()), view<2>(z.get<1>())));

    Tuple<Tuple<V>, Tuple<V, V, V>> zz = y;
    REQUIRE(rs::equal(x.get<0>(), zz.get<0>()));
    REQUIRE(rs::equal(view<0>(x.get<1>()), view<0>(zz.get<1>())));
    REQUIRE(rs::equal(view<1>(x.get<1>()), view<1>(zz.get<1>())));
    REQUIRE(rs::equal(view<2>(x.get<1>()), view<2>(zz.get<1>())));
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
    REQUIRE(rs::equal(view<1>(y), std::vector{4, 5, 6}));
    y = 0;
    REQUIRE(rs::equal(view<0>(y), zeros));
    REQUIRE(rs::equal(view<1>(y), zeros));
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
    REQUIRE(rs::equal(view<1>(y), w));
    y = 0;
    REQUIRE(rs::equal(view<0>(y), zeros));
    REQUIRE(rs::equal(view<1>(y), zeros));
    REQUIRE(rs::equal(view<1>(y), w));
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
    REQUIRE(rs::equal(view<0>(y), std::vector{1, 4, 9}));
    REQUIRE(rs::equal(view<1>(y), std::vector{16, 25}));
    REQUIRE(rs::equal(view<2>(y), std::vector{16, 9, 4, 1}));

    auto z = Tuple<std::vector<int>, std::vector<int>, std::vector<int>>{};
    z = y | vs::transform([](auto&& i) { return i + i; });
    REQUIRE(rs::equal(view<0>(z), std::vector{2, 8, 18}));
    REQUIRE(rs::equal(view<1>(z), std::vector{32, 50}));
    REQUIRE(rs::equal(view<2>(z), std::vector{32, 18, 8, 2}));
}

TEST_CASE("Pipe syntax for Non-Owning ThreeTuples")
{
    using namespace ccs;
    using namespace field::tuple;

    auto x = Tuple{std::vector{1, 2, 3}, std::vector{4, 5}, std::vector{4, 3, 2, 1}};
    auto y = x | vs::transform([](auto&& i) { return i * i; });
    REQUIRE(traits::ThreeTuple<decltype(y)>);
    REQUIRE(rs::equal(view<0>(y), std::vector{1, 4, 9}));
    REQUIRE(rs::equal(view<1>(y), std::vector{16, 25}));
    REQUIRE(rs::equal(view<2>(y), std::vector{16, 9, 4, 1}));

    auto a = std::vector<int>(3);
    auto b = std::vector<int>(2);
    auto c = std::vector<int>(4);
    auto z = Tuple{a, b, c};
    REQUIRE(traits::Non_Tuple_Input_Range<decltype(
                view<0>(y | vs::transform([](auto&& i) { return i + i; })))>);
    z = y | vs::transform([](auto&& i) { return i + i; });
    REQUIRE(rs::equal(view<0>(z), std::vector{2, 8, 18}));
    REQUIRE(rs::equal(view<1>(z), std::vector{32, 50}));
    REQUIRE(rs::equal(view<2>(z), std::vector{32, 18, 8, 2}));
}

TEST_CASE("ThreeTuples with ThreeTuplePipes")
{
    using namespace ccs;
    using namespace field::tuple;

    auto x = Tuple{std::vector{1, 2, 3}, std::vector{4, 5}, std::vector{4, 3, 2, 1}};
    auto y = x | Tuple{vs::transform([](auto&& i) { return i * i; }),
                       vs::transform([](auto&& i) { return i + i; }),
                       vs::transform([](auto&& i) { return i * i * i; })};
    REQUIRE(traits::ThreeTuple<decltype(y)>);
    REQUIRE(rs::equal(view<0>(y), std::vector{1, 4, 9}));
    REQUIRE(rs::equal(view<1>(y), std::vector{8, 10}));
    REQUIRE(rs::equal(view<2>(y), std::vector{64, 27, 8, 1}));

    auto a = std::vector<int>(3);
    auto b = std::vector<int>(2);
    auto c = std::vector<int>(4);
    auto z = Tuple{a, b, c};
    REQUIRE(traits::Non_Tuple_Input_Range<decltype(
                view<0>(y | vs::transform([](auto&& i) { return i + i; })))>);
    z = y | vs::transform([](auto&& i) { return i + i; });
    REQUIRE(rs::equal(view<0>(z), std::vector{2, 8, 18}));
    REQUIRE(rs::equal(view<1>(z), std::vector{16, 20}));
    REQUIRE(rs::equal(view<2>(z), std::vector{128, 54, 16, 2}));

    auto q = Tuple<std::span<int>, std::span<int>, std::span<int>>{a, b, c};

    q | Tuple{vs::transform([](auto&& i) { return i * i; }),
              vs::transform([](auto&& i) { return i + i; }),
              vs::transform([](auto&& i) { return i * i * i; })};

    q = y | Tuple{vs::transform([](auto&& i) { return i * i; }),
                  vs::transform([](auto&& i) { return i + i; }),
                  vs::transform([](auto&& i) { return i * i * i; })};

    REQUIRE(rs::equal(a, std::vector{1, 16, 81}));
    REQUIRE(rs::equal(b, std::vector{16, 20}));
    REQUIRE(rs::equal(c, std::vector{262144, 19683, 512, 1}));

    vs::transform([](auto&& i) { return i; }) |
        Tuple{vs::transform([](auto&& i) { return i * i; }),
              vs::transform([](auto&& i) { return i + i; }),
              vs::transform([](auto&& i) { return i * i * i; })};
    // vs::transform([](auto&& i) { return i; }) |
    //     Tuple{vs::transform([](auto&& i) { return 2 * i; })};
}
