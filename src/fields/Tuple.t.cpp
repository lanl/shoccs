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

#if 0

TEST_CASE("directional")
{
    using namespace ccs;
    using T = std::vector<real>;

    auto r = vs::iota(0, 16);
    auto x = directional_field<T, 0>{vs::iota(0, 16), int3{2, 4, 2}};

    REQUIRE(x[1] == 1);
    REQUIRE(rs::equal(x, r));

    {
        auto y = directional_field<T, 0>{int3{2, 4, 2}};
        REQUIRE(y.size() == 16u);
        REQUIRE(y.extents() == int3{2, 4, 2});
    }

    auto y = x + x;
    REQUIRE(y[1] == 2);
    REQUIRE(y.extents() == int3{2, 4, 2});
    REQUIRE(rs::equal(y, vs::zip_with(std::plus{}, r, r)));

    auto z = 1 + y + 2;
    REQUIRE(z[0] == 3);
    REQUIRE(z.extents() == int3{2, 4, 2});
    REQUIRE(rs::equal(z, vs::zip_with(std::plus{}, y, vs::repeat(3))));

    auto yy = directional_field<T, 0>{};
    yy = z;
    REQUIRE(z.extents() == yy.extents());
    REQUIRE(rs::equal(z, yy));

    auto u = z + r_tuple{vs::iota(16, 32)};
    REQUIRE(u[0] == 19);
    REQUIRE(u.extents() == int3{2, 4, 2});
    REQUIRE(rs::equal(u, vs::zip_with(std::plus{}, z, vs::iota(16, 32))));

    auto v = r_tuple{vs::iota(16, 32)} + z;
    REQUIRE(rs::equal(u, v));

    auto xx = directional_field<T, 0>{r_tuple{vs::iota(16, 32)} + z};
}

template <int I>
using D = ccs::directional_field<std::vector<ccs::real>, I>;
TEST_CASE("tuple of directional")
{
    using namespace ccs;
    constexpr int3 extents{2, 4, 2};

    SECTION("1 tuple")
    {
        using N = r_tuple<D<0>>;

        REQUIRE(!All<D<0>>);
        REQUIRE(All<D<0>&>);

        {
            N x = r_tuple{D<0>{vs::iota(0, 16), extents}};
            REQUIRE(traits::Owning_R_Tuple<decltype(x)>);

            N y{extents};
            REQUIRE(y.size() == 16u);
            REQUIRE(y.get<0>().extents() == extents);
        }

        auto d = D<0>{vs::iota(0, 16), extents};
        REQUIRE(d[0] == 0);
        N x = r_tuple{MOVE(d)};
        const auto& v = x.get<0>();
        REQUIRE(x[0] == 0);
        REQUIRE(v[0] == 0);
        REQUIRE(v[15] == 15);

        auto y = (x + 1) + 0;

        {
            const auto& z = y.get<0>();
            REQUIRE(rs::equal(z, vs::iota(1, 17)));
            REQUIRE(z.extents() == extents);
        }

        {
            N xx{y};
            const auto& z = xx.get<0>();
            REQUIRE(rs::equal(z, vs::iota(1, 17)));
            REQUIRE(z.extents() == extents);
        }

        {
            N xx{};
            xx = y;
            const auto& z = xx.get<0>();
            REQUIRE(rs::equal(z, vs::iota(1, 17)));
            REQUIRE(z.extents() == extents);
        }
    }

    SECTION("3 tuple")
    {
        using N = r_tuple<D<0>, D<1>, D<2>>;

        N x{D<0>{vs::iota(0, 16), extents},
            D<1>{vs::iota(16, 32), extents},
            D<2>{vs::iota(32, 48), extents}};

        REQUIRE(rs::equal(x.get<0>(), vs::iota(0, 16)));
        REQUIRE(rs::equal(x.get<2>(), vs::iota(32, 48)));

        auto y = (x + 0) + 1 + 0;
        REQUIRE(traits::Owning_R_Tuple<decltype(y)>);
        REQUIRE(traits::Directional_Field<decltype(y.get<0>())>);
        REQUIRE(!traits::Owning_R_Tuple<decltype(y.get<0>().as_r_tuple())>);

        REQUIRE(rs::equal(y.get<0>(), vs::iota(1, 17)));
        REQUIRE(rs::equal(y.get<1>(), vs::iota(17, 33)));
        REQUIRE(rs::equal(y.get<2>(), vs::iota(33, 49)));

        REQUIRE(y.get<0>().extents() == extents);
        REQUIRE(y.get<1>().extents() == extents);
        REQUIRE(y.get<2>().extents() == extents);

        {
            N z{y};
            REQUIRE(rs::equal(z.get<0>(), vs::iota(1, 17)));
            REQUIRE(rs::equal(z.get<1>(), vs::iota(17, 33)));
            REQUIRE(rs::equal(z.get<2>(), vs::iota(33, 49)));

            REQUIRE(z.get<0>().extents() == extents);
            REQUIRE(z.get<1>().extents() == extents);
            REQUIRE(z.get<2>().extents() == extents);
        }

        {
            N z{};
            z = y;
            REQUIRE(rs::equal(z.get<0>(), vs::iota(1, 17)));
            REQUIRE(rs::equal(z.get<1>(), vs::iota(17, 33)));
            REQUIRE(rs::equal(z.get<2>(), vs::iota(33, 49)));

            REQUIRE(z.get<0>().extents() == extents);
            REQUIRE(z.get<1>().extents() == extents);
            REQUIRE(z.get<2>().extents() == extents);
        }

        {
            N q{extents, extents, extents};
            REQUIRE(q.get<0>().extents() == extents);
            REQUIRE(q.get<1>().extents() == extents);
            REQUIRE(q.get<2>().extents() == extents);
        }
    }
}

TEST_CASE("nested tuple of directional")
{
    using namespace ccs;
    constexpr int3 extents{2, 4, 2};

    SECTION("1 tuple")
    {
        using N = r_tuple<r_tuple<D<0>>>;

        {
            N x{detail::tag(), extents};
        }

        N x{r_tuple{D<0>{vs::iota(0, 16), extents}}};
        REQUIRE(traits::Owning_R_Tuple<decltype(x)>);
        auto& x0 = x.get<0>();
        REQUIRE(traits::Owning_R_Tuple<decltype(x0)>);
        auto& x1 = x0.get<0>();
        REQUIRE(traits::Directional_Field<decltype(x1)>);
        REQUIRE(traits::Owning_R_Tuple<decltype(x1.as_r_tuple())>);

        const auto& v = x.get<0>().get<0>();
        REQUIRE(x[0] == 0);
        REQUIRE(v[0] == 0);
        REQUIRE(v[15] == 15);

        auto y = (x + 1);
        REQUIRE(traits::Owning_R_Tuple<decltype(y)>);
        auto& y0 = y.get<0>();
        REQUIRE(traits::Owning_R_Tuple<decltype(y0)>);
        REQUIRE(traits::Directional_Field<decltype(y0.get<0>())>);
        REQUIRE(!traits::Owning_R_Tuple<decltype(y0.get<0>().as_r_tuple())>);

        {
            const auto& z = y.get<0>().get<0>();
            REQUIRE(rs::equal(z, vs::iota(1, 17)));
            REQUIRE(z.extents() == extents);
        }

        {
            N xx{y};
            const auto& z = xx.get<0>().get<0>();
            REQUIRE(rs::equal(z, vs::iota(1, 17)));
            REQUIRE(z.extents() == extents);
        }

        {
            N xx{};
            xx = y;
            const auto& z = xx.get<0>().get<0>();
            REQUIRE(rs::equal(z, vs::iota(1, 17)));
            REQUIRE(z.extents() == extents);
        }
    }

    SECTION("3 tuple")
    {
        using M = r_tuple<D<0>, D<1>, D<2>>;
        using N = r_tuple<M>;

        {
            N x{detail::tag(), r_tuple{extents, extents, extents}};
        }

        N x{r_tuple{D<0>{vs::iota(0, 16), extents},
                    D<1>{vs::iota(16, 32), extents},
                    D<2>{vs::iota(32, 48), extents}}};

        REQUIRE(traits::Owning_R_Tuple<decltype(x)>);
        auto& x0 = x.get<0>();
        REQUIRE(traits::Owning_R_Tuple<decltype(x0)>);
        auto& x1 = x0.get<2>();
        REQUIRE(traits::Directional_Field<decltype(x1)>);
        REQUIRE(traits::Owning_R_Tuple<decltype(x1.as_r_tuple())>);

        auto y = (x + 0) + 1 + 0;
        REQUIRE(traits::Owning_R_Tuple<decltype(y)>);
        const auto& y0 = y.get<0>();
        REQUIRE(traits::Owning_R_Tuple<decltype(y0)>);
        REQUIRE(traits::Directional_Field<decltype(y0.get<1>())>);
        REQUIRE(!traits::Owning_R_Tuple<decltype(y0.get<1>().as_r_tuple())>);

        {
            N z{y};
            const auto& zz = z.get<0>();
            REQUIRE(rs::equal(zz.get<0>(), vs::iota(1, 17)));
            REQUIRE(rs::equal(zz.get<1>(), vs::iota(17, 33)));
            REQUIRE(rs::equal(zz.get<2>(), vs::iota(33, 49)));

            REQUIRE(zz.get<0>().extents() == extents);
            REQUIRE(zz.get<1>().extents() == extents);
            REQUIRE(zz.get<2>().extents() == extents);
        }

        {
            N z{};
            z = y;
            const auto& zz = z.get<0>();
            REQUIRE(rs::equal(zz.get<0>(), vs::iota(1, 17)));
            REQUIRE(rs::equal(zz.get<1>(), vs::iota(17, 33)));
            REQUIRE(rs::equal(zz.get<2>(), vs::iota(33, 49)));

            REQUIRE(zz.get<0>().extents() == extents);
            REQUIRE(zz.get<1>().extents() == extents);
            REQUIRE(zz.get<2>().extents() == extents);
        }
    }
}

TEST_CASE("Scalar Directional")
{
    using namespace ccs;
    using T = std::vector<real>;

    constexpr int3 extents{4, 2, 2};

    // default construction
    scalar_directional<T, 0> x{};

    // sized construction
    {
        auto y = scalar_y<T>{r_tuple{extents}, r_tuple{16, 17, 18}};
        REQUIRE(y.get<0>().get<0>().extents() == extents);
        REQUIRE(view<0>(y.get<1>()).size() == 16u);
        REQUIRE(view<1>(y.get<1>()).size() == 17u);
        REQUIRE(view<2>(y.get<1>()).size() == 18u);
    }

    auto y = scalar_directional<T, 0>{
        r_tuple{directional_field<T, 0>{vs::iota(0, 16), extents}},
        r_tuple{vs::iota(0, 16), vs::iota(16, 32), vs::iota(32, 48)}};

    x = y;

    REQUIRE(rs::equal(vs::iota(16, 32), x.get<1>().get<1>()));
    REQUIRE(rs::equal(vs::iota(32, 48), view<2>(x.get<1>())));

    auto z = (0 + x) + 1;
    REQUIRE(traits::Owning_R_Tuple<decltype(z)>);

    scalar_x<T> zz{z};

    {
        auto& z0 = z.get<0>();
        REQUIRE(traits::Owning_R_Tuple<decltype(z0)>);
        auto& d = z0.get<0>();
        REQUIRE(traits::Directional_Field<decltype(d)>);
        REQUIRE(!traits::Owning_R_Tuple<decltype(d.as_r_tuple())>);
        REQUIRE(rs::equal(d, vs::iota(1, 17)));
        REQUIRE(d.extents() == extents);
    }

    {
        const auto& z1 = z.get<1>();
        REQUIRE(!traits::Owning_R_Tuple<decltype(z1)>);
        REQUIRE(rs::equal(view<0>(z1), vs::iota(1, 17)));
        REQUIRE(rs::equal(view<1>(z1), vs::iota(17, 33)));
        REQUIRE(rs::equal(view<2>(z1), vs::iota(33, 49)));
    }
}

TEST_CASE("Scalar")
{
    using namespace ccs;
    using T = std::vector<real>;

    constexpr int3 extents{4, 2, 2};

    // default construction
    scalar<T> x{};

    // sized construction
    {
        auto y = scalar<T>{r_tuple{extents, extents, extents}, r_tuple{24, 26, 28}};
    }

    auto y = scalar<T>{r_tuple{directional_field<T, 0>{vs::iota(0, 16), extents},
                               directional_field<T, 1>{vs::iota(16, 32), extents},
                               directional_field<T, 2>{vs::iota(32, 48), extents}},
                       r_tuple{vs::iota(0, 16), vs::iota(16, 32), vs::iota(32, 48)}};

    x = y;

    REQUIRE(rs::equal(vs::iota(16, 32), x.get<1>().get<1>()));
    REQUIRE(rs::equal(vs::iota(32, 48), view<2>(x.get<1>())));

    auto z = (0 + x) + 1 + y;
    REQUIRE(traits::Owning_R_Tuple<decltype(z)>);

    {
        auto& z0 = z.get<0>();
        REQUIRE(traits::Owning_R_Tuple<decltype(z0)>);
        const auto& a = z0.get<0>();
        const auto& b = z0.get<1>();
        const auto& c = z0.get<2>();
        REQUIRE(traits::Directional_Field<decltype(a)>);
        REQUIRE(!traits::Owning_R_Tuple<decltype(a.as_r_tuple())>);
        REQUIRE(
            rs::equal(a, vs::zip_with(std::plus{}, vs::iota(1, 17), vs::iota(0, 16))));
        REQUIRE(
            rs::equal(b, vs::zip_with(std::plus{}, vs::iota(17, 33), vs::iota(16, 32))));
        REQUIRE(
            rs::equal(c, vs::zip_with(std::plus{}, vs::iota(33, 49), vs::iota(32, 48))));

        REQUIRE(a.extents() == extents);
        REQUIRE(b.extents() == extents);
        REQUIRE(c.extents() == extents);
    }

    {
        scalar<T> zz{z};
        const auto& z1 = zz.get<1>();
        REQUIRE(traits::Owning_R_Tuple<decltype(z1)>);
        REQUIRE(rs::equal(view<0>(z1),
                          vs::zip_with(std::plus{}, vs::iota(1, 17), vs::iota(0, 16))));
        REQUIRE(rs::equal(view<1>(z1),
                          vs::zip_with(std::plus{}, vs::iota(17, 33), vs::iota(16, 32))));
        REQUIRE(rs::equal(view<2>(z1),
                          vs::zip_with(std::plus{}, vs::iota(33, 49), vs::iota(32, 48))));
    }
}
#endif