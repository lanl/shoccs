#include "Location.hpp"
#include "selector.hpp"
//#include "views.hpp"

#include <catch2/catch_test_macros.hpp>
#include <range/v3/algorithm/equal.hpp>
#include <range/v3/view/cartesian_product.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/repeat_n.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip_with.hpp>

using namespace ccs;
using namespace si;

constexpr auto plus = lift(std::plus{});

// construct simple mesh geometry
namespace global
{

const auto x = std::vector<real>{0, 1, 2, 3, 4};
const auto y = std::vector<real>{-2, -1};
const auto z = std::vector<real>{6, 7, 8, 9};
const auto rx = std::vector<real3>{{0.5, -2, 6}, {1.5, -1, 9}};
const auto ry = std::vector<real3>{{1, -1.75, 7}, {4, -1.25, 7}, {3, -1.1, 9}};
const auto rz = std::vector<real3>{{0, -2, 6.1}};
const auto loc = location{x, y, z, rx, ry, rz};

} // namespace global

TEST_CASE("construction")
{
    using T = std::vector<int>;
    scalar<T> s{tuple{vs::iota(0, 10)},
                tuple{vs::iota(0, 3), vs::iota(3, 6), vs::iota(-1, 2)}};

    auto&& [D, Rxyz] = s;
    REQUIRE(D == vs::iota(0, 10));
    REQUIRE(Rxyz == tuple{vs::iota(0, 3), vs::iota(3, 6), vs::iota(-1, 2)});

    static_assert(std::same_as<decltype(get<0, 0>(s)), decltype(get<0>(get<0>(s)))>);
    static_assert(std::same_as<decltype(get<si::D>(s)), decltype(get<0>(get<0>(s)))>);
}

TEST_CASE("conversion")
{
    using T = std::vector<int>;

    auto s = scalar<T>{tuple{std::vector{1, 2, 3}},
                       tuple{std::vector{1}, std::vector{2, 3}, std::vector{4, 5, 6}}};

    scalar<std::span<const int>> r = s;

    REQUIRE(r == s);

    auto f = [](scalar<std::span<int>> x) { x *= 2; };
    s = f;
    REQUIRE(r == scalar<T>{tuple{T{2, 4, 6}}, tuple{T{2}, T{4, 6}, T{8, 10, 12}}});
}

TEST_CASE("selection")
{
    // some initialization
    auto v = std::vector{1, 2};
    auto rx = std::vector{3, 4};
    auto ry = std::vector{5, 6};
    auto rz = std::vector{7, 8};
    auto s = scalar<std::vector<int>&>(tuple{v}, tuple{rx, ry, rz});

    // Add tests to verify the type of the containers and views.
    // THey may point to scalar needing to be a tuple of nested tuples
    REQUIRE(rs::equal(v, s | sel::D));
    REQUIRE(rs::equal(rx, s | sel::Rx));
    REQUIRE(rs::equal(ry, s | sel::Ry));
    REQUIRE(rs::equal(rz, s | sel::Rz));

    // can compose these with pipe syntax since selections are just view closures
    constexpr auto t = sel::D | vs::transform([](auto&& x) { return x * x * x; });

    REQUIRE(rs::equal(s | t, std::vector{1, 8}));

    // modify selection
    s | sel::D = 0;
    REQUIRE(rs::equal(v, vs::repeat_n(0, 2)));

    s | sel::Rz = -1;
    REQUIRE(rs::equal(rz, vs::repeat_n(-1, 2)));

    s | sel::R = 2;
    REQUIRE(rs::equal(v, vs::repeat_n(0, 2)));
    REQUIRE(rs::equal(rx, vs::repeat_n(2, 2)));
}

TEST_CASE("math")
{
    auto s = scalar<std::vector<int>>{
        tuple{vs::iota(1, 5)}, tuple{vs::iota(6, 10), vs::iota(6, 12), vs::iota(10, 15)}};

    auto q = s + 1;
    REQUIRE(rs::equal(vs::iota(2, 6), get<0>(q)));
    REQUIRE(rs::equal(vs::iota(7, 11), get<Rx>(q)));
    REQUIRE(rs::equal(vs::iota(7, 13), get<Ry>(q)));
    REQUIRE(rs::equal(vs::iota(11, 16), get<Rz>(q)));
    auto r = q + s;

    REQUIRE(r == tuple{tuple{plus(vs::iota(2, 6), vs::iota(1, 5))},
                       tuple{plus(vs::iota(6, 10), vs::iota(7, 11)),
                             plus(vs::iota(6, 12), vs::iota(7, 13)),
                             plus(vs::iota(10, 15), vs::iota(11, 16))}});

    scalar<std::vector<int>> t = r;
    REQUIRE(t == r);

    scalar<std::vector<int>> a{s};
    scalar<std::span<int>> b = a;
    b = r;
    REQUIRE(b == r);
}

TEST_CASE("lifting single arg")
{
    auto s = scalar<std::vector<int>>{
        tuple{vs::iota(1, 5)}, tuple{vs::iota(6, 10), vs::iota(6, 12), vs::iota(10, 15)}};

    constexpr auto f = lift([](auto&& x) { return std::abs(x) + 1; });

    auto j = f(s);
    auto k = s + 1;

    REQUIRE(j == k);
}

TEST_CASE("lifting multiple args")
{
    auto x = scalar<std::vector<int>>{
        tuple{vs::iota(1, 5)},
        tuple{vs::iota(6, 10), vs::iota(6, 12), std::vector{-4, 5, -10, 6}}};
    auto y = scalar<std::vector<int>>{
        tuple{vs::iota(2, 6)},
        tuple{vs::iota(5, 9), vs::iota(-6, 0), std::vector{10, -8, 2, -7}}};

    constexpr auto f =
        lift([](auto&& x, auto&& y) { return std::max(std::abs(x), std::abs(y)); });

    auto z = f(x, y);

    REQUIRE(rs::equal(get<0>(z), vs::iota(2, 6)));
    REQUIRE(rs::equal(get<Rx>(z), vs::iota(6, 10)));
    REQUIRE(rs::equal(get<Ry>(z), vs::iota(6, 12)));
    REQUIRE(rs::equal(get<Rz>(z), std::vector{10, 8, 10, 7}));
}

TEST_CASE("mesh location vector")
{
    using namespace global;

    auto s = scalar<std::vector<real>>{tuple{x.size() * y.size() * z.size()},
                                       tuple{rx.size(), ry.size(), rz.size()}};

    constexpr auto t = vs::transform([](auto&& loc) {
        auto&& [x, y, z] = loc;
        return x * y * z;
    });

    auto sol = loc.view() | t;

    REQUIRE(rs::equal(s | sel::D | sol, vs::cartesian_product(x, y, z) | t));
    REQUIRE(rs::equal(s | sel::Rx | sol, rx | t));
    REQUIRE(rs::equal(s | sel::Ry | sol, ry | t));
    REQUIRE(rs::equal(s | sel::Rz | sol, rz | t));
    REQUIRE(rs::equal(get<0>(s | sel::R | sol), rx | t));
    REQUIRE(rs::equal(get<1>(s | sel::R | sol), ry | t));
    REQUIRE(rs::equal(get<2>(s | sel::R | sol), rz | t));

    s | sel::Rx = sol;
    REQUIRE(rs::equal(get<Rx>(s), rx | t));

    s | sel::Ry = sol;
    REQUIRE(rs::equal(get<Ry>(s), ry | t));

    s | sel::Rz = sol;
    REQUIRE(rs::equal(get<Rz>(s), rz | t));

    constexpr auto u = vs::transform([](auto&& loc) {
        auto&& [x, y, z] = loc;
        return x * y * y * z;
    });
    // the problem with this statement has to with the assignability
    // of the view_tuples components
    s | sel::R = loc.view() | u;
    REQUIRE(rs::equal(s | sel::Rx, rx | u));
    REQUIRE(rs::equal(s | sel::Ry, ry | u));
    REQUIRE(rs::equal(s | sel::Rz, rz | u));
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
    using namespace global;

    auto t = std::vector<real>(x.size() * y.size() * z.size());
    auto u = std::vector<real>(rx.size());
    auto v = std::vector<real>(ry.size());
    auto w = std::vector<real>(rz.size());

    auto s = scalar<std::span<real>>{tuple{t}, tuple{u, v, w}};

    constexpr auto tr = vs::transform([](auto&& loc) {
        auto&& [x, y, z] = loc;
        return x * y * z;
    });

    constexpr auto sol = loc.view() | tr;

    s | sel::D = sol;
    REQUIRE(rs::equal(s | sel::D, vs::cartesian_product(x, y, z) | tr));
    REQUIRE(rs::equal(s | sel::Rx, vs::repeat_n(0.0, rx.size())));

    s | sel::Rx = sol;
    REQUIRE(rs::equal(s | sel::Rx, rx | tr));

    s | sel::Ry = sol;
    REQUIRE(rs::equal(s | sel::Ry, ry | tr));

    s | sel::Rz = sol;
    REQUIRE(rs::equal(s | sel::Rz, rz | tr));

    constexpr auto ur = vs::transform([](auto&& loc) {
        auto&& [x, y, z] = loc;
        return x * y * y * z;
    });

    s | sel::R = loc.view() | ur;
    REQUIRE(rs::equal(s | sel::Rx, rx | ur));
    REQUIRE(rs::equal(s | sel::Ry, ry | ur));
    REQUIRE(rs::equal(s | sel::Rz, rz | ur));

    static_assert(ViewClosures<decltype(loc.view() | tuple{vs::transform(loc_fn<0>{}),
                                                           vs::transform(loc_fn<1>{}),
                                                           vs::transform(loc_fn<2>{})})>);
    s | sel::R = loc.view() | tuple{vs::transform(loc_fn<0>{}),
                                    vs::transform(loc_fn<1>{}),
                                    vs::transform(loc_fn<2>{})};

    REQUIRE(rs::equal(u, rx | vs::transform(loc_fn<0>{})));
    REQUIRE(rs::equal(v, ry | vs::transform(loc_fn<1>{})));
    REQUIRE(rs::equal(w, rz | vs::transform(loc_fn<2>{})));
}
