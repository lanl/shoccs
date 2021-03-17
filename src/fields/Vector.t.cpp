#include "Scalar.hpp"
#include "Vector.hpp"

#include "Location.hpp"
#include "Selector.hpp"

#include <catch2/catch_test_macros.hpp>
#include <range/v3/algorithm/equal.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/transform.hpp>

using namespace ccs::selector::vector;
using namespace ccs::selector::scalar;

// construct simple mesh geometry
namespace global
{

const auto x = std::vector<ccs::real>{0, 1, 2, 3, 4};
const auto y = std::vector<ccs::real>{-2, -1};
const auto z = std::vector<ccs::real>{6, 7, 8, 9};
const auto rx = std::vector<ccs::real3>{{0.5, -2, 6}, {1.5, -1, 9}};
const auto ry = std::vector<ccs::real3>{{1, -1.75, 7}, {4, -1.25, 7}, {3, -1.1, 9}};
const auto rz = std::vector<ccs::real3>{{0, -2, 6.1}};
const auto loc = ccs::field::tuple::Location{x, y, z, rx, ry, rz};

} // namespace global

TEST_CASE("construction")
{
    using namespace ccs;
    using namespace field;
    using T = std::vector<int>;

    Vector<T> v{Tuple{Tuple{vs::iota(0, 10)},
                      Tuple{vs::iota(0, 3), vs::iota(1, 3), vs::iota(-1, 1)}},
                Tuple{Tuple{vs::iota(10, 20)},
                      Tuple{vs::iota(1, 4), vs::iota(2, 5), vs::iota(2, 10)}},
                Tuple{Tuple{vs::iota(20, 30)},
                      Tuple{vs::iota(1, 5), vs::iota(5, 10), vs::iota(8, 10)}}};

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
    using namespace ccs;
    using namespace ccs::field;

    using T = std::vector<int>;

    Vector<T> v{Tuple{Tuple{vs::iota(0, 10)},
                      Tuple{vs::iota(0, 3), vs::iota(1, 3), vs::iota(-1, 1)}},
                Tuple{Tuple{vs::iota(10, 20)},
                      Tuple{vs::iota(1, 4), vs::iota(2, 5), vs::iota(2, 10)}},
                Tuple{Tuple{vs::iota(20, 30)},
                      Tuple{vs::iota(1, 5), vs::iota(5, 10), vs::iota(8, 10)}}};

    Vector<std::span<const int>> r = v;

    REQUIRE(get<Dx>(v).size() == get<Dx>(r).size());
    REQUIRE(rs::equal(get<Dx>(v), get<Dx>(r)));

    // auto f = [](SimpleVector<std::span<int>> x) {
    //     auto q = SimpleVector<std::vector<int>>{
    //         &global::loc,
    //         Tuple{std::vector{1, 2, 3}, std::vector{4, 5, 6}, std::vector{7, 8, 9}},
    //         Tuple{std::vector{1}, std::vector{2, 3}, std::vector{4, 5, 6}}};
    //     x = 2 * q;
    // };

    // s = f;
    // REQUIRE(rs::equal(s | selector::Dz, std::vector{14, 16, 18}));
    // REQUIRE(rs::equal(s | selector::Rx, std::vector{2}));
    // REQUIRE(rs::equal(s | selector::Ry, std::vector{4, 6}));
    // REQUIRE(rs::equal(s | selector::Rz, std::vector{8, 10, 12}));
}

TEST_CASE("to scalar")
{
    using namespace ccs;
    using namespace ccs::field;
    using T = std::vector<real>;

    Vector<T> v{Tuple{Tuple{vs::iota(0, 10)},
                      Tuple{vs::iota(0, 3), vs::iota(1, 3), vs::iota(-1, 1)}},
                Tuple{Tuple{vs::iota(10, 20)},
                      Tuple{vs::iota(1, 4), vs::iota(2, 5), vs::iota(2, 10)}},
                Tuple{Tuple{vs::iota(20, 30)},
                      Tuple{vs::iota(1, 5), vs::iota(5, 10), vs::iota(8, 10)}}};

    {
        ScalarView_Const sx = get<X>(v);
        REQUIRE(rs::equal(get<D>(sx), get<Dx>(v)));
        REQUIRE(rs::equal(get<Rx>(sx), get<xRx>(v)));
    }

    {
        ScalarView_Mutable sx = get<X>(v);
        sx | selector::D = 1;
        REQUIRE(rs::equal(get<Dx>(v), vs::repeat_n(1., 10)));
    }

    {
        ScalarView_Const sy = get<Y>(v);
        REQUIRE(rs::equal(sy | selector::D, get<Dy>(v)));
        REQUIRE(rs::equal(sy | selector::Ry, get<yRy>(v)));
    }

    {
        ScalarView_Mutable sy = get<Y>(v);
        sy | selector::D = 1;
        REQUIRE(rs::equal(get<Dy>(v), vs::repeat_n(1., 10)));
    }

    {
        ScalarView_Const sz = get<Z>(v);
        REQUIRE(rs::equal(sz | selector::D, get<Dz>(v)));
        REQUIRE(rs::equal(sz | selector::Rz, get<zRz>(v)));
    }

    {
        ScalarView_Mutable sz = get<Z>(v);
        sz | selector::D = 1;
        REQUIRE(rs::equal(get<Dz>(v), vs::repeat_n(1., 10)));
    }

    {
        VectorView_Const vvc = v;
        ScalarView_Const sx = get<X>(vvc);
        REQUIRE(rs::equal(sx | selector::D, get<Dx>(vvc)));
    }
}

TEST_CASE("selection")
{
    using namespace ccs;
    using namespace field;

    using T = std::vector<real>;

    Vector<T> v_{Tuple{Tuple{vs::iota(0, 10)},
                       Tuple{vs::iota(0, 3), vs::iota(1, 3), vs::iota(-1, 1)}},
                 Tuple{Tuple{vs::iota(10, 20)},
                       Tuple{vs::iota(1, 4), vs::iota(2, 5), vs::iota(2, 10)}},
                 Tuple{Tuple{vs::iota(20, 30)},
                       Tuple{vs::iota(1, 5), vs::iota(5, 10), vs::iota(8, 10)}}};
    VectorView_Mutable v = v_;

    REQUIRE(rs::equal(vs::iota(0, 10), v | selector::Dx));
    REQUIRE(rs::equal(vs::iota(10, 20), v | selector::Dy));
    REQUIRE(rs::equal(vs::iota(20, 30), v | selector::Dz));
    REQUIRE(rs::equal(vs::iota(0, 3), get<X>(v | selector::Rx)));
    REQUIRE(rs::equal(vs::iota(2, 5), get<Y>(v | selector::Ry)));
    REQUIRE(rs::equal(vs::iota(8, 10), get<Z>(v | selector::Rz)));

    // modify selection
    v | selector::D = 0;
    REQUIRE(rs::equal(get<Dx>(v_), vs::repeat_n(0, 10)));
    REQUIRE(rs::equal(get<Dy>(v_), vs::repeat_n(0, 10)));
    REQUIRE(rs::equal(get<Dz>(v_), vs::repeat_n(0, 10)));

    v | selector::Rz = -1;
    REQUIRE(rs::equal(get<xRz>(v_), vs::repeat_n(-1, 2)));
    REQUIRE(rs::equal(get<yRz>(v_), vs::repeat_n(-1, 8)));
    REQUIRE(rs::equal(get<zRz>(v_), vs::repeat_n(-1, 2)));

    v | selector::R = 2;
    REQUIRE(rs::equal(get<xRx>(v_), vs::repeat_n(2, 3)));
    REQUIRE(rs::equal(get<xRz>(v_), vs::repeat_n(2, 2)));
    REQUIRE(rs::equal(get<yRx>(v_), vs::repeat_n(2, 3)));
}

TEST_CASE("math")
{
    using namespace ccs;
    using namespace ccs::field;
    using T = std::vector<int>;

    Vector<T> v{Tuple{Tuple{vs::iota(0, 10)},
                      Tuple{vs::iota(0, 3), vs::iota(1, 3), vs::iota(-1, 1)}},
                Tuple{Tuple{vs::iota(10, 20)},
                      Tuple{vs::iota(1, 4), vs::iota(2, 5), vs::iota(2, 10)}},
                Tuple{Tuple{vs::iota(20, 30)},
                      Tuple{vs::iota(1, 5), vs::iota(5, 10), vs::iota(8, 10)}}};

    auto q = v + 1;
    REQUIRE(rs::equal(vs::iota(1, 11), q | selector::Dx));
    REQUIRE(rs::equal(vs::iota(11, 21), q | selector::Dy));
    REQUIRE(rs::equal(vs::iota(21, 31), q | selector::Dz));
    REQUIRE(rs::equal(vs::iota(1, 4), get<xRx>(q)));
    REQUIRE(rs::equal(vs::iota(3, 6), get<yRy>(q)));
    REQUIRE(rs::equal(vs::iota(9, 11), get<zRz>(q)));

    auto r = q + v;
    constexpr auto plus = [](auto&& a, auto&& b) {
        return vs::zip_with(std::plus{}, FWD(a), FWD(b));
    };
    REQUIRE(rs::equal(plus(vs::iota(1, 11), vs::iota(0, 10)), r | selector::Dx));
    REQUIRE(rs::equal(plus(vs::iota(2, 6), vs::iota(1, 5)), get<zRx>(r)));

    Vector<std::vector<int>> t = r;
    REQUIRE(rs::equal(t | selector::Dz, r | selector::Dz));
    REQUIRE(rs::equal(t | selector::yRy, r | selector::yRy));

    Vector<std::vector<int>> a{v};
    Vector<std::span<int>> b = a;
    b = r;
    REQUIRE(rs::equal(b | selector::Dy, r | selector::Dy));
    REQUIRE(rs::equal(b | selector::zRx, r | selector::zRx));
}

TEST_CASE("lifting single arg")
{
    using namespace ccs;
    using namespace ccs::field;
    using T = std::vector<int>;

    Vector<T> v{Tuple{Tuple{vs::iota(0, 10)},
                      Tuple{vs::iota(0, 3), vs::iota(1, 3), vs::iota(-1, 1)}},
                Tuple{Tuple{vs::iota(10, 20)},
                      Tuple{vs::iota(1, 4), vs::iota(2, 5), vs::iota(2, 10)}},
                Tuple{Tuple{vs::iota(20, 30)},
                      Tuple{vs::iota(1, 5), vs::iota(5, 10), vs::iota(8, 10)}}};

    constexpr auto f = tuple::lift([](auto&& x) { return std::abs(x) + 1; });

    auto j = f(v);
    auto k = v + 1;

    REQUIRE(rs::equal(j | selector::Dx, k | selector::Dx));
    REQUIRE(rs::equal(j | selector::Dy, k | selector::Dy));
    REQUIRE(rs::equal(j | selector::Dz, k | selector::Dz));
    REQUIRE(rs::equal(j | selector::xRx, k | selector::xRx));
    REQUIRE(rs::equal(j | selector::yRy, k | selector::yRy));
    REQUIRE(rs::equal(j | selector::zRz, k | selector::zRz));
}

TEST_CASE("mesh location")
{
    using namespace ccs;
    using namespace ccs::field;
    using namespace global;

    using T = std::vector<real>;

    const auto sz = vs::repeat_n(0, x.size() * y.size() * z.size());
    const auto x_sz = vs::repeat_n(0, rx.size());
    const auto y_sz = vs::repeat_n(0, ry.size());
    const auto z_sz = vs::repeat_n(0, rz.size());

    Vector<T> v{Tuple{Tuple{sz}, Tuple{x_sz, y_sz, z_sz}},
                Tuple{Tuple{sz}, Tuple{x_sz, y_sz, z_sz}},
                Tuple{Tuple{sz}, Tuple{x_sz, y_sz, z_sz}}};

    constexpr auto t = vs::transform([](auto&& loc) {
        auto&& [x, y, z] = loc;
        return x * y * z;
    });

    constexpr auto sol = loc.view() | t;

    v | selector::D = sol;
    REQUIRE(rs::equal(v | selector::Dx, vs::cartesian_product(x, y, z) | t));

    v | selector::Rx = sol;
    REQUIRE(rs::equal(v | selector::xRx, rx | t));
    REQUIRE(rs::equal(v | selector::yRx, rx | t));

    v | selector::Ry = sol;
    REQUIRE(rs::equal(v | selector::xRy, ry | t));
    REQUIRE(rs::equal(v | selector::zRy, ry | t));

    v | selector::Rz = sol;
    REQUIRE(rs::equal(v | selector::zRz, rz | t));

    constexpr auto u = vs::transform([](auto&& loc) {
        auto&& [x, y, z] = loc;
        return x * y * y * z;
    });

    v | selector::R = loc.view() | u;
    REQUIRE(rs::equal(v | selector::xRx, rx | u));
    REQUIRE(rs::equal(v | selector::yRy, ry | u));
    REQUIRE(rs::equal(v | selector::zRz, rz | u));
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
    using namespace ccs;
    using namespace ccs::field;
    using namespace global;

    using T = std::vector<real>;

    const auto sz = vs::repeat_n(0, x.size() * y.size() * z.size());
    const auto x_sz = vs::repeat_n(0, rx.size());
    const auto y_sz = vs::repeat_n(0, ry.size());
    const auto z_sz = vs::repeat_n(0, rz.size());

    Vector<T> v_{Tuple{Tuple{sz}, Tuple{x_sz, y_sz, z_sz}},
                 Tuple{Tuple{sz}, Tuple{x_sz, y_sz, z_sz}},
                 Tuple{Tuple{sz}, Tuple{x_sz, y_sz, z_sz}}};
    VectorView_Mutable v = v_;

    constexpr auto t = vs::transform([](auto&& loc) {
        auto&& [x, y, z] = loc;
        return x * y * z;
    });

    constexpr auto sol = loc.view() | t;

    v | selector::D = sol;
    REQUIRE(rs::equal(v | selector::Dx, vs::cartesian_product(x, y, z) | t));

    v | selector::Rx = sol;
    REQUIRE(rs::equal(v | selector::xRx, rx | t));
    REQUIRE(rs::equal(v | selector::yRx, rx | t));

    v | selector::Ry = sol;
    REQUIRE(rs::equal(v | selector::xRy, ry | t));
    REQUIRE(rs::equal(v | selector::zRy, ry | t));

    v | selector::Rz = sol;
    REQUIRE(rs::equal(v | selector::zRz, rz | t));

    constexpr auto u = vs::transform([](auto&& loc) {
        auto&& [x, y, z] = loc;
        return x * y * y * z;
    });

    v | selector::R = loc.view() | u;
    REQUIRE(rs::equal(v | selector::xRx, rx | u));
    REQUIRE(rs::equal(v | selector::yRy, ry | u));
    REQUIRE(rs::equal(v | selector::zRz, rz | u));

    v | selector::Rx = loc.view() | Tuple{vs::transform(loc_fn<0>{}),
                                          vs::transform(loc_fn<1>{}),
                                          vs::transform(loc_fn<2>{})};

    REQUIRE(rs::equal(v | selector::xRx, rx | vs::transform(loc_fn<0>{})));
    REQUIRE(rs::equal(v | selector::yRx, rx | vs::transform(loc_fn<1>{})));
    REQUIRE(rs::equal(v | selector::zRx, rx | vs::transform(loc_fn<2>{})));
}
