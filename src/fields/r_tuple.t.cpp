#include "r_tuple.hpp"

#include "types.hpp"

#include <catch2/catch_test_macros.hpp>

#include <range/v3/algorithm/equal.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/zip_with.hpp>

#include <vector>

#include <iostream>

TEST_CASE("concepts")
{
    using namespace ccs;
    using T = std::vector<real>;

    REQUIRE(All<T&>);
    REQUIRE(!All<T>);

    REQUIRE(!All<detail::container_tuple<T>&>);
    REQUIRE(!All<detail::container_tuple<T>>);

    
    REQUIRE(All<r_tuple<T>&>);
    REQUIRE(!All<r_tuple<T>>);

}

TEST_CASE("construction")
{
    using namespace ccs;

    auto x = std::vector<real>{0, 1, 2};
    auto y = std::vector<real>{3, 4, 5};
    auto z = std::vector<real>{6, 7, 8};

    auto r = r_tuple(x);
    REQUIRE(std::same_as<decltype(r), r_tuple<std::vector<real>&>>);
    REQUIRE(r[1] == 1);
    r[1] = -1;
    REQUIRE(r[1] == -1);

    
    REQUIRE(rs::equal(view<0>(r), x));

    SECTION("owning from input range")
    {
        using T = r_tuple<std::vector<real>>;
        REQUIRE(std::same_as<decltype(std::declval<T>().template get<0>()),
                             std::vector<real>&>);
        // try different sizes to flush out memory issues
        for (int i = 10; i < 1024; i += 9) {
            auto xx = r_tuple<std::vector<real>>{vs::iota(0, i)};
            REQUIRE(rs::equal(vs::iota(0, i), xx));

            xx = vs::iota(0, i / 2);
            REQUIRE(rs::equal(xx, vs::iota(0, i / 2)));
        }
    }

    SECTION("owning from non-owning")
    {
        auto r_owning = r_tuple{std::vector<real>{0, 1, 2}};
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
            r_tuple{std::vector<real>{}, std::vector<real>{}, std::vector<real>{}};
        auto a = r_tuple{x, y, z};

        r_owning = a;
        REQUIRE(rs::equal(view<0>(r_owning), x));
        REQUIRE(rs::equal(view<1>(r_owning), y));
        REQUIRE(rs::equal(view<2>(r_owning), z));
    }

    auto rr = r_tuple(x, y, z);
    REQUIRE(rr.N == 3);
    REQUIRE(rs::equal(view<0>(rr), x));
    REQUIRE(rs::equal(view<1>(rr), y));
    REQUIRE(rs::equal(view<2>(rr), z));

    {
        auto xx = x;
        auto yy = y;
        auto zz = z;
        auto r_owning = r_tuple{MOVE(xx), MOVE(yy), MOVE(zz)};
        REQUIRE(r_owning.N == 3);
        REQUIRE(rs::equal(view<0>(rr), view<0>(r_owning)));
        REQUIRE(rs::equal(view<1>(rr), view<1>(r_owning)));
        REQUIRE(rs::equal(view<2>(rr), view<2>(r_owning)));
    }
}

TEST_CASE("container math")
{
    using namespace ccs;
    using T = std::vector<real>;

    auto x = r_tuple<T>{vs::iota(0, 10)};
    x += 5;
    REQUIRE(rs::equal(x, vs::iota(5, 15)));

    r_tuple<T> y = r_tuple{vs::iota(0, 10)};
    y += 5;
    REQUIRE(rs::equal(x, y));

    auto xx = r_tuple<T, T>{vs::iota(-5, 3), vs::iota(-10, 1)};
    auto yy = r_tuple{vs::iota(-5, 3), vs::iota(-10, 1)};
    xx += 100;
    REQUIRE(rs::equal(view<0>(xx), vs::iota(95, 103)));
    REQUIRE(rs::equal(view<1>(xx), vs::iota(90, 101)));

    xx += yy;
    REQUIRE(rs::equal(view<0>(xx),
                      vs::zip_with(std::plus{}, vs::iota(95, 103), vs::iota(-5, 3))));
    REQUIRE(rs::equal(view<1>(xx),
                      vs::zip_with(std::plus{}, vs::iota(90, 101), vs::iota(-10, 1))));
}

TEST_CASE("view math with numeric")
{
    using namespace ccs;

    using T = std::vector<real>;

    for (int i = 100; i < 1000; i += 10) {
        {
            auto x = r_tuple<T>{vs::iota(-i, i)};
            auto z = 5 + x + 5;

            REQUIRE(rs::equal(z, vs::iota(10 - i, 10 + i)));
            REQUIRE(rs::equal(z, vs::iota(10 - i, 10 + i)));
        }

        {
            auto vx = vs::iota(-i, 0);
            auto vy = vs::iota(0, i);
            auto vz = vs::iota(-1, 2 * i);
            auto xx = r_tuple<T, T, T>{vx, vy, vz};
            auto zz = (xx + 5) + 1;

            auto o = vs::repeat(6);

            REQUIRE(rs::equal(view<0>(zz), vs::zip_with(std::plus{}, vx, o)));
            REQUIRE(rs::equal(view<1>(zz), vs::zip_with(std::plus{}, vy, o)));
            REQUIRE(rs::equal(view<2>(zz), vs::zip_with(std::plus{}, vz, o)));
        }

        {
            auto x = vs::iota(0, i);
            auto y = vs::iota(i, 2 * i);
            auto z = vs::iota(2 * i, 3 * i);
            auto xyz = r_tuple<T, T, T>(x, y, z);

            auto s = xyz + r_tuple{x};

            REQUIRE(rs::equal(view<0>(s), vs::zip_with(std::plus{}, x, x)));
            REQUIRE(rs::equal(view<1>(s), vs::zip_with(std::plus{}, y, x)));
            REQUIRE(rs::equal(view<2>(s), vs::zip_with(std::plus{}, z, x)));
        }
    }
}

TEST_CASE("view math with tuples")
{
    using namespace ccs;

    using T = std::vector<real>;

    auto x = r_tuple<T>{vs::iota(0, 10)};
    auto y = r_tuple{vs::iota(10, 20)};
    auto z = x + y;

    REQUIRE(rs::equal(z, vs::zip_with(std::plus{}, vs::iota(0, 10), vs::iota(10, 20))));

    auto g = z + x + x + y;
    auto h = z + z + x;

    REQUIRE(g.size() == h.size());
    REQUIRE(rs::equal(g, h));

    {
        auto xx = r_tuple<T>{x + y};
        REQUIRE(rs::equal(z, xx));

        auto yy = r_tuple<T>{};
        yy = x + y;
        REQUIRE(rs::equal(yy, xx));
    }

    {
        auto xx = r_tuple<T, T, T>{x + y, z + x + x + y, z + z + x};
        REQUIRE(rs::equal(view<0>(xx), z));
        REQUIRE(rs::equal(view<1>(xx), g));
        REQUIRE(rs::equal(view<2>(xx), h));

        auto yy = r_tuple<T, T, T>{};
        yy = r_tuple{x + y, z + x + x + y, z + z + x};
        REQUIRE(rs::equal(view<0>(yy), z));
        REQUIRE(rs::equal(view<1>(yy), g));
        REQUIRE(rs::equal(view<2>(yy), h));
    }
}

TEST_CASE("directional")
{
    using namespace ccs;
    using T = std::vector<real>;

    auto r = vs::iota(0, 16);
    auto x = directional_field<T, 0>{vs::iota(0, 16), int3{2, 4, 2}};

    REQUIRE(x[1] == 1);
    REQUIRE(rs::equal(x, r));

    auto y = x + x;
    REQUIRE(y[1] == 2);
    REQUIRE(y.extents() == int3{2, 4, 2});
    REQUIRE(rs::equal(y, vs::zip_with(std::plus{}, r, r)));

    auto z = 1 + y + 2;
    REQUIRE(z[0] == 3);
    REQUIRE(z.extents() == int3{2, 4, 2});
    REQUIRE(rs::equal(z, vs::zip_with(std::plus{}, y, vs::repeat(3))));

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
TEST_CASE("directional_composite")
{
    using namespace ccs;
    constexpr int3 extents{2, 4, 2};

    SECTION("1 tuple")
    {
        using N = r_tuple<D<0>>;

        REQUIRE(!All<D<0>>);
        REQUIRE(All<D<0>&>);
        auto d = D<0>{vs::iota(0, 16), extents};
        REQUIRE(d[0] == 0);
        N x = r_tuple{MOVE(d)};
        const auto& v = x.get<0>();
        REQUIRE(x[0] == 0);
        REQUIRE(v[0] == 0);
        REQUIRE(v[15] == 15);

        //auto y = (x + 1) + 0;
        //const auto& z = y.get<0>();
        //REQUIRE(rs::equal(z, vs::iota(1, 17)));
        //REQUIRE(z.extents() == extents);
    }
#if 0
    SECTION("3 tuple")
    {
        using N = r_tuple<D<0>, D<1>, D<2>>;

        N x{D<0>{vs::iota(0, 16), extents},
            D<1>{vs::iota(16, 32), extents},
            D<2>{vs::iota(32, 48), extents}};

        REQUIRE(rs::equal(x.get<0>(), vs::iota(0, 16)));
        REQUIRE(rs::equal(x.get<2>(), vs::iota(32, 48)));

        auto y = (x + 0) + 1 + 0;

        REQUIRE(rs::equal(y.get<0>(), vs::iota(1, 17)));
        REQUIRE(rs::equal(y.get<1>(), vs::iota(17, 33)));
        REQUIRE(rs::equal(y.get<2>(), vs::iota(33, 49)));

        REQUIRE(y.get<0>().extents() == extents);
        REQUIRE(y.get<1>().extents() == extents);
        REQUIRE(y.get<2>().extents() == extents);

        // N z;
        // z = y;
    }
#endif
}