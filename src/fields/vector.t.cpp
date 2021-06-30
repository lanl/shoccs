#include "vector.hpp"
#include "scalar.hpp"

#include "selector.hpp"

#include <catch2/catch_test_macros.hpp>
#include <range/v3/all.hpp>

using namespace ccs;
using namespace si;
using namespace vi;

constexpr auto plus = lift(std::plus{});

// construct simple mesh geometry
namespace g
{

const auto x = std::vector<real>{0, 1, 2, 3, 4};
const auto y = std::vector<real>{-2, -1};
const auto z = std::vector<real>{6, 7, 8, 9};
const auto rx = std::vector<real3>{{0.5, -2, 6}, {1.5, -1, 9}};
const auto ry = std::vector<real3>{{1, -1.75, 7}, {4, -1.25, 7}, {3, -1.1, 9}};
const auto rz = std::vector<real3>{{0, -2, 6.1}};

const auto sloc = tuple{tuple{vs::cartesian_product(x, y, z)},
                        tuple{vs::all(rx), vs::all(ry), vs::all(rz)}};
const auto loc = tuple{sloc, sloc, sloc};
} // namespace g

TEST_CASE("construction")
{
    using T = std::vector<int>;

    vector<T> v{tuple{tuple{vs::iota(0, 10)},
                      tuple{vs::iota(0, 3), vs::iota(1, 3), vs::iota(-1, 1)}},
                tuple{tuple{vs::iota(10, 20)},
                      tuple{vs::iota(1, 4), vs::iota(2, 5), vs::iota(2, 10)}},
                tuple{tuple{vs::iota(20, 30)},
                      tuple{vs::iota(1, 5), vs::iota(5, 10), vs::iota(8, 10)}}};

    auto&& [X, Y, Z] = v;
    REQUIRE(rs::equal(get<0>(X), vs::iota(0, 10)));
    REQUIRE(rs::equal(get<0>(Y), vs::iota(10, 20)));
    REQUIRE(rs::equal(get<0>(Z), vs::iota(20, 30)));

    REQUIRE(rs::equal(get<Dx>(v), vs::iota(0, 10)));
    REQUIRE(rs::equal(get<xRx>(v), vs::iota(0, 3)));
    REQUIRE(rs::equal(get<xRy>(v), vs::iota(1, 3)));
    REQUIRE(rs::equal(get<xRz>(v), vs::iota(-1, 1)));
    REQUIRE(rs::equal(get<yRx>(v), vs::iota(1, 4)));
    REQUIRE(rs::equal(get<yRy>(v), vs::iota(2, 5)));
    REQUIRE(rs::equal(get<yRz>(v), vs::iota(2, 10)));
    REQUIRE(rs::equal(get<zRx>(v), vs::iota(1, 5)));
    REQUIRE(rs::equal(get<zRy>(v), vs::iota(5, 10)));
    REQUIRE(rs::equal(get<zRz>(v), vs::iota(8, 10)));
}

TEST_CASE("conversion")
{
    using T = std::vector<int>;

    vector<T> v{tuple{tuple{vs::iota(0, 10)},
                      tuple{vs::iota(0, 3), vs::iota(1, 3), vs::iota(-1, 1)}},
                tuple{tuple{vs::iota(10, 20)},
                      tuple{vs::iota(1, 4), vs::iota(2, 5), vs::iota(2, 10)}},
                tuple{tuple{vs::iota(20, 30)},
                      tuple{vs::iota(1, 5), vs::iota(5, 10), vs::iota(8, 10)}}};

    vector<std::span<const int>> r = v;
    REQUIRE(r == v);

    // auto f = [](Simplevector<std::span<int>> x) {
    //     auto q = Simplevector<std::vector<int>>{
    //         &global::loc,
    //         tuple{std::vector{1, 2, 3}, std::vector{4, 5, 6}, std::vector{7, 8, 9}},
    //         tuple{std::vector{1}, std::vector{2, 3}, std::vector{4, 5, 6}}};
    //     x = 2 * q;
    // };

    // s = f;
    // REQUIRE(rs::equal(s | sel::Dz, std::vector{14, 16, 18}));
    // REQUIRE(rs::equal(s | sel::Rx, std::vector{2}));
    // REQUIRE(rs::equal(s | sel::Ry, std::vector{4, 6}));
    // REQUIRE(rs::equal(s | sel::Rz, std::vector{8, 10, 12}));
}

TEST_CASE("to scalar")
{
    using T = std::vector<real>;

    vector<T> v{tuple{tuple{vs::iota(0, 10)},
                      tuple{vs::iota(0, 3), vs::iota(1, 3), vs::iota(-1, 1)}},
                tuple{tuple{vs::iota(10, 20)},
                      tuple{vs::iota(1, 4), vs::iota(2, 5), vs::iota(2, 10)}},
                tuple{tuple{vs::iota(20, 30)},
                      tuple{vs::iota(1, 5), vs::iota(5, 10), vs::iota(8, 10)}}};

    {
        scalar_view sx = get<X>(v);
        REQUIRE(sx == get<0>(v));
    }

    {
        scalar_span sx = get<X>(v);
        REQUIRE(sx == get<0>(v));
        sx | sel::D = 1;
        REQUIRE(rs::equal(get<Dx>(v), vs::repeat_n(1., 10)));
    }

    {
        scalar_view sy = get<Y>(v);
        REQUIRE(sy == get<1>(v));
    }

    {
        scalar_span sy = get<Y>(v);
        REQUIRE(sy == get<1>(v));
        sy | sel::D = 1;
        REQUIRE(rs::equal(get<Dy>(v), vs::repeat_n(1., 10)));
    }

    {
        scalar_view sz = get<Z>(v);
        REQUIRE(sz == get<2>(v));
    }

    {
        scalar_span sz = get<Z>(v);
        REQUIRE(sz == get<2>(v));
        sz | sel::D = 1;
        REQUIRE(rs::equal(get<Dz>(v), vs::repeat_n(1., 10)));
    }

    {
        vector_view vvc = v;
        scalar_view sx = get<X>(vvc);
        REQUIRE(sx == get<0>(vvc));
    }
}

TEST_CASE("selection")
{
    using T = std::vector<real>;

    vector<T> v_{tuple{tuple{vs::iota(0, 10)},
                       tuple{vs::iota(0, 3), vs::iota(1, 3), vs::iota(-1, 1)}},
                 tuple{tuple{vs::iota(10, 20)},
                       tuple{vs::iota(1, 4), vs::iota(2, 5), vs::iota(2, 10)}},
                 tuple{tuple{vs::iota(20, 30)},
                       tuple{vs::iota(1, 5), vs::iota(5, 10), vs::iota(8, 10)}}};
    vector_span v = v_;

    REQUIRE(rs::equal(v | sel::Dx, vs::iota(0, 10)));
    REQUIRE(rs::equal(vs::iota(10, 20), v | sel::Dy));
    REQUIRE(rs::equal(vs::iota(20, 30), v | sel::Dz));
    REQUIRE(rs::equal(vs::iota(0, 3), get<X>(v | sel::Rx)));
    REQUIRE(rs::equal(vs::iota(2, 5), get<Y>(v | sel::Ry)));
    REQUIRE(rs::equal(vs::iota(8, 10), get<Z>(v | sel::Rz)));

    // modify selection
    v | sel::D = 0;
    REQUIRE(rs::equal(get<Dx>(v_), vs::repeat_n(0, 10)));
    REQUIRE(rs::equal(get<Dy>(v_), vs::repeat_n(0, 10)));
    REQUIRE(rs::equal(get<Dz>(v_), vs::repeat_n(0, 10)));

    v | sel::Rz = -1;
    REQUIRE(rs::equal(get<xRz>(v_), vs::repeat_n(-1, 2)));
    REQUIRE(rs::equal(get<yRz>(v_), vs::repeat_n(-1, 8)));
    REQUIRE(rs::equal(get<zRz>(v_), vs::repeat_n(-1, 2)));

    v | sel::R = 2;
    REQUIRE(rs::equal(get<xRx>(v_), vs::repeat_n(2, 3)));
    REQUIRE(rs::equal(get<xRz>(v_), vs::repeat_n(2, 2)));
    REQUIRE(rs::equal(get<yRx>(v_), vs::repeat_n(2, 3)));
}

TEST_CASE("math")
{
    using T = std::vector<int>;

    vector<T> v{tuple{tuple{vs::iota(0, 10)},
                      tuple{vs::iota(0, 3), vs::iota(1, 3), vs::iota(-1, 1)}},
                tuple{tuple{vs::iota(10, 20)},
                      tuple{vs::iota(1, 4), vs::iota(2, 5), vs::iota(2, 10)}},
                tuple{tuple{vs::iota(20, 30)},
                      tuple{vs::iota(1, 5), vs::iota(5, 10), vs::iota(8, 10)}}};

    auto q = v + 1;
    REQUIRE(rs::equal(vs::iota(1, 11), q | sel::Dx));
    REQUIRE(rs::equal(vs::iota(11, 21), q | sel::Dy));
    REQUIRE(rs::equal(vs::iota(21, 31), q | sel::Dz));
    REQUIRE(rs::equal(vs::iota(1, 4), get<xRx>(q)));
    REQUIRE(rs::equal(vs::iota(3, 6), get<yRy>(q)));
    REQUIRE(rs::equal(vs::iota(9, 11), get<zRz>(q)));

    auto r = q + v;

    REQUIRE(rs::equal(plus(vs::iota(1, 11), vs::iota(0, 10)), r | sel::Dx));
    REQUIRE(rs::equal(plus(vs::iota(2, 6), vs::iota(1, 5)), get<zRx>(r)));

    vector<T> t = r;
    REQUIRE(rs::equal(t | sel::Dz, r | sel::Dz));
    REQUIRE(rs::equal(t | sel::yRy, r | sel::yRy));

    vector<T> a{v};
    vector<std::span<int>> b = a;
    b = r;
    REQUIRE(rs::equal(b | sel::Dy, r | sel::Dy));
    REQUIRE(rs::equal(b | sel::zRx, r | sel::zRx));
}

TEST_CASE("lifting single arg")
{
    using T = std::vector<int>;

    vector<T> v{tuple{tuple{vs::iota(0, 10)},
                      tuple{vs::iota(0, 3), vs::iota(1, 3), vs::iota(-1, 1)}},
                tuple{tuple{vs::iota(10, 20)},
                      tuple{vs::iota(1, 4), vs::iota(2, 5), vs::iota(2, 10)}},
                tuple{tuple{vs::iota(20, 30)},
                      tuple{vs::iota(1, 5), vs::iota(5, 10), vs::iota(8, 10)}}};

    constexpr auto f = lift([](auto&& x) { return std::abs(x) + 1; });

    auto j = f(v);
    auto k = v + 1;

    REQUIRE(rs::equal(j | sel::Dx, k | sel::Dx));
    REQUIRE(rs::equal(j | sel::Dy, k | sel::Dy));
    REQUIRE(rs::equal(j | sel::Dz, k | sel::Dz));
    REQUIRE(rs::equal(j | sel::xRx, k | sel::xRx));
    REQUIRE(rs::equal(j | sel::yRy, k | sel::yRy));
    REQUIRE(rs::equal(j | sel::zRz, k | sel::zRz));
}

TEST_CASE("mesh location")
{
    using namespace g;

    using T = std::vector<real>;

    const auto sz = vs::repeat_n(0, x.size() * y.size() * z.size());
    const auto x_sz = vs::repeat_n(0, rx.size());
    const auto y_sz = vs::repeat_n(0, ry.size());
    const auto z_sz = vs::repeat_n(0, rz.size());

    vector<T> v{tuple{tuple{sz}, tuple{x_sz, y_sz, z_sz}},
                tuple{tuple{sz}, tuple{x_sz, y_sz, z_sz}},
                tuple{tuple{sz}, tuple{x_sz, y_sz, z_sz}}};

    constexpr auto t = vs::transform([](auto&& loc) {
        auto&& [x, y, z] = loc;
        return x * y * z;
    });

    const auto sol = loc | t;

    v | sel::D = sol;
    REQUIRE(rs::equal(v | sel::Dx, vs::cartesian_product(x, y, z) | t));

    v | sel::Rx = sol;
    REQUIRE(rs::equal(v | sel::xRx, rx | t));
    REQUIRE(rs::equal(v | sel::yRx, rx | t));

    v | sel::Ry = sol;
    REQUIRE(rs::equal(v | sel::xRy, ry | t));
    REQUIRE(rs::equal(v | sel::zRy, ry | t));

    v | sel::Rz = sol;
    REQUIRE(rs::equal(v | sel::zRz, rz | t));

    constexpr auto u = vs::transform([](auto&& loc) {
        auto&& [x, y, z] = loc;
        return x * y * y * z;
    });

    v | sel::R = loc | u;
    REQUIRE(rs::equal(v | sel::xRx, rx | u));
    REQUIRE(rs::equal(v | sel::yRy, ry | u));
    REQUIRE(rs::equal(v | sel::zRz, rz | u));
}

template <auto I>
struct loc_fn {
    constexpr auto operator()(auto&& loc)
    {
        auto&& [x, y, z] = loc;
        return x * y * z * std::get<I>(loc);
    }
};

TEST_CASE("mesh location span")
{
    using namespace g;

    using T = std::vector<real>;

    const auto sz = vs::repeat_n(0, x.size() * y.size() * z.size());
    const auto x_sz = vs::repeat_n(0, rx.size());
    const auto y_sz = vs::repeat_n(0, ry.size());
    const auto z_sz = vs::repeat_n(0, rz.size());

    vector<T> v_{tuple{tuple{sz}, tuple{x_sz, y_sz, z_sz}},
                 tuple{tuple{sz}, tuple{x_sz, y_sz, z_sz}},
                 tuple{tuple{sz}, tuple{x_sz, y_sz, z_sz}}};
    vector_span v = v_;

    constexpr auto t = vs::transform([](auto&& loc) {
        auto&& [x, y, z] = loc;
        return x * y * z;
    });

    const auto sol = loc | t;

    REQUIRE(Vector<decltype(sol)>);

    v | sel::D = sol;
    REQUIRE(rs::equal(v | sel::Dx, vs::cartesian_product(x, y, z) | t));

    v | sel::Rx = sol;
    REQUIRE(rs::equal(v | sel::xRx, rx | t));
    REQUIRE(rs::equal(v | sel::yRx, rx | t));

    v | sel::Ry = sol;
    REQUIRE(rs::equal(v | sel::xRy, ry | t));
    REQUIRE(rs::equal(v | sel::zRy, ry | t));

    v | sel::Rz = sol;
    REQUIRE(rs::equal(v | sel::zRz, rz | t));

    constexpr auto u = vs::transform([](auto&& loc) {
        auto&& [x, y, z] = loc;
        return x * y * y * z;
    });

    v | sel::R = loc | u;
    REQUIRE(rs::equal(v | sel::xRx, rx | u));
    REQUIRE(rs::equal(v | sel::yRy, ry | u));
    REQUIRE(rs::equal(v | sel::zRz, rz | u));

    REQUIRE(Vector<decltype(loc | tuple{vs::transform(loc_fn<0>{}),
                                        vs::transform(loc_fn<1>{}),
                                        vs::transform(loc_fn<2>{})})>);

    v | sel::Rx = loc | sel::Rx |
                  tuple{vs::transform(loc_fn<0>{}),
                        vs::transform(loc_fn<1>{}),
                        vs::transform(loc_fn<2>{})};

    REQUIRE(rs::equal(v | sel::xRx, rx | vs::transform(loc_fn<0>{})));
    REQUIRE(rs::equal(v | sel::yRx, rx | vs::transform(loc_fn<1>{})));
    REQUIRE(rs::equal(v | sel::zRx, rx | vs::transform(loc_fn<2>{})));
}
