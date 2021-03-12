#include "Scalar.hpp"

#include "Location.hpp"
#include "Selector.hpp"
//#include "views.hpp"

#include <catch2/catch_test_macros.hpp>
#include <range/v3/algorithm/equal.hpp>
#include <range/v3/view/cartesian_product.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/repeat_n.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip_with.hpp>

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
    Scalar<T> s{Tuple{vs::iota(0, 10)},
                Tuple{vs::iota(0, 3), vs::iota(3, 6), vs::iota(-1, 2)}};

    auto&& [D, Rxyz] = s;
    REQUIRE(rs::equal(get<0>(D), vs::iota(0, 10)));
    REQUIRE(rs::equal(get<0>(Rxyz), vs::iota(0, 3)));
    REQUIRE(rs::equal(get<1>(Rxyz), vs::iota(3, 6)));
    REQUIRE(rs::equal(get<2>(Rxyz), vs::iota(-1, 2)));
}

TEST_CASE("conversion")
{
    using namespace ccs;
    using namespace field;

    auto s = Scalar<std::vector<int>>{
        Tuple{std::vector{1, 2, 3}},
        Tuple{std::vector{1}, std::vector{2, 3}, std::vector{4, 5, 6}}};

    Scalar<std::span<const int>> r = s;

    REQUIRE(get<0>(s).size() == get<0>(r).size());
    REQUIRE(rs::equal(get<0>(s), get<0>(r)));

    // auto f = [](SimpleScalar<std::span<int>> x) {
    //     auto q = SimpleScalar<std::vector<int>>{
    //         &global::loc,
    //         Tuple{std::vector{1, 2, 3}},
    //         Tuple{std::vector{1}, std::vector{2, 3}, std::vector{4, 5, 6}}};
    //     x = 2 * q;
    // };

    // s = f;
    // REQUIRE(rs::equal(s | selector::D, std::vector{2, 4, 6}));
    // REQUIRE(rs::equal(s | selector::Rx, std::vector{2}));
    // REQUIRE(rs::equal(s | selector::Ry, std::vector{4, 6}));
    // REQUIRE(rs::equal(s | selector::Rz, std::vector{8, 10, 12}));
}

TEST_CASE("selection")
{
    using namespace ccs;
    using namespace field;
    // some initialization
    auto v = std::vector{1, 2};
    auto rx = std::vector{3, 4};
    auto ry = std::vector{5, 6};
    auto rz = std::vector{7, 8};
    auto s = Scalar<std::vector<int>&>(Tuple{v}, Tuple{rx, ry, rz});

    // Add tests to verify the type of the containers and views.
    // THey may point to scalar needing to be a tuple of nested tuples
    REQUIRE(rs::equal(v, s | selector::D));
    REQUIRE(rs::equal(rx, s | selector::Rx));
    REQUIRE(rs::equal(ry, s | selector::Ry));
    REQUIRE(rs::equal(rz, s | selector::Rz));

    REQUIRE(rs::equal(s | selector::D | vs::transform([](auto&& x) { return x * x * x; }),
                      std::vector{1, 8}));

    // modify selection
    s | selector::D = 0;
    REQUIRE(rs::equal(v, vs::repeat_n(0, 2)));

    s | selector::Rz = -1;
    REQUIRE(rs::equal(rz, vs::repeat_n(-1, 2)));

    s | selector::R = 2;
    REQUIRE(rs::equal(v, vs::repeat_n(0, 2)));
    REQUIRE(rs::equal(rx, vs::repeat_n(2, 2)));
}

TEST_CASE("math")
{
    using namespace ccs;
    using namespace field;
    auto s = Scalar<std::vector<int>>{
        Tuple{vs::iota(1, 5)}, Tuple{vs::iota(6, 10), vs::iota(6, 12), vs::iota(10, 15)}};

    auto q = s + 1;
    REQUIRE(rs::equal(vs::iota(2, 6), get<0>(q)));
    REQUIRE(rs::equal(vs::iota(7, 11), get<0, 1>(q)));
    REQUIRE(rs::equal(vs::iota(7, 13), get<1, 1>(q)));
    REQUIRE(rs::equal(vs::iota(11, 16), get<2, 1>(q)));
    auto r = q + s;
    constexpr auto plus = [](auto&& a, auto&& b) {
        return vs::zip_with(std::plus{}, FWD(a), FWD(b));
    };
    REQUIRE(rs::equal(plus(vs::iota(2, 6), vs::iota(1, 5)), get<0>(r)));
    REQUIRE(rs::equal(plus(vs::iota(11, 16), vs::iota(10, 15)), get<2, 1>(r)));

    Scalar<std::vector<int>> t = r;
    REQUIRE(rs::equal(get<0>(t), get<0>(r)));
    REQUIRE(rs::equal(get<0, 1>(t), get<0, 1>(r)));
    REQUIRE(rs::equal(get<1, 1>(t), get<1, 1>(r)));
    REQUIRE(rs::equal(get<2, 1>(t), get<2, 1>(r)));

    Scalar<std::vector<int>> a{s};
    Scalar<std::span<int>> b = a;
    b = r;
    REQUIRE(rs::equal(get<0>(b), get<0>(r)));
    REQUIRE(rs::equal(get<0, 1>(b), get<0, 1>(r)));
    REQUIRE(rs::equal(get<1, 1>(b), get<1, 1>(r)));
    REQUIRE(rs::equal(get<2, 1>(b), get<2, 1>(r)));
}
TEST_CASE("lifting single arg")
{
    using namespace ccs;
    using namespace ccs::field;
    auto s = Scalar<std::vector<int>>{
        Tuple{vs::iota(1, 5)}, Tuple{vs::iota(6, 10), vs::iota(6, 12), vs::iota(10, 15)}};

    constexpr auto f = tuple::lift([](auto&& x) { return std::abs(x) + 1; });

    auto j = f(s);
    auto k = s + 1;

    REQUIRE(rs::equal(get<0>(j), get<0>(k)));
    REQUIRE(rs::equal(get<0, 1>(j), get<0, 1>(k)));
    REQUIRE(rs::equal(get<1, 1>(j), get<1, 1>(k)));
    REQUIRE(rs::equal(get<2, 1>(j), get<2, 1>(k)));
}

TEST_CASE("lifting multiple args")
{
    using namespace ccs;
    using namespace ccs::field;
    auto x = Scalar<std::vector<int>>{
        Tuple{vs::iota(1, 5)},
        Tuple{vs::iota(6, 10), vs::iota(6, 12), std::vector{-4, 5, -10, 6}}};
    auto y = Scalar<std::vector<int>>{
        Tuple{vs::iota(2, 6)},
        Tuple{vs::iota(5, 9), vs::iota(-6, 0), std::vector{10, -8, 2, -7}}};

    constexpr auto f = tuple::lift(
        [](auto&& x, auto&& y) { return std::max(std::abs(x), std::abs(y)); });

    auto z = f(x, y);

    REQUIRE(rs::equal(get<0>(z), vs::iota(2, 6)));
    REQUIRE(rs::equal(get<0, 1>(z), vs::iota(6, 10)));
    REQUIRE(rs::equal(get<1, 1>(z), vs::iota(6, 12)));
    REQUIRE(rs::equal(get<2, 1>(z), std::vector{10, 8, 10, 7}));
}

TEST_CASE("mesh location vector")
{
    using namespace ccs;
    using namespace ccs::field;
    using namespace global;

    auto s = Scalar<std::vector<real>>{Tuple{x.size() * y.size() * z.size()},
                                       Tuple{rx.size(), ry.size(), rz.size()}};

    constexpr auto t = vs::transform([](auto&& loc) {
        auto&& [x, y, z] = loc;
        return x * y * z;
    });

    auto sol = loc.view() | t;

    REQUIRE(rs::equal(s | selector::D | sol, vs::cartesian_product(x, y, z) | t));
    REQUIRE(rs::equal(s | selector::Rx | sol, rx | t));
    REQUIRE(rs::equal(s | selector::Ry | sol, ry | t));
    REQUIRE(rs::equal(s | selector::Rz | sol, rz | t));
    REQUIRE(rs::equal(get<0>(s | selector::R | sol), rx | t));
    REQUIRE(rs::equal(get<1>(s | selector::R | sol), ry | t));
    REQUIRE(rs::equal(get<2>(s | selector::R | sol), rz | t));

    s | selector::Rx = sol;
    REQUIRE(rs::equal(get<0, 1>(s), rx | t));

    s | selector::Ry = sol;
    REQUIRE(rs::equal(get<1, 1>(s), ry | t));

    s | selector::Rz = sol;
    REQUIRE(rs::equal(get<2, 1>(s), rz | t));

    constexpr auto u = vs::transform([](auto&& loc) {
        auto&& [x, y, z] = loc;
        return x * y * y * z;
    });

    s | selector::R = loc.view() | u;
    REQUIRE(rs::equal(s | selector::Rx, rx | u));
    REQUIRE(rs::equal(s | selector::Ry, ry | u));
    REQUIRE(rs::equal(s | selector::Rz, rz | u));
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

    auto t = std::vector<real>(x.size() * y.size() * z.size());
    auto u = std::vector<real>(rx.size());
    auto v = std::vector<real>(ry.size());
    auto w = std::vector<real>(rz.size());

    // auto s = Scalar<std::span<real>>{
    //     Tuple{std::span<real>(t)},
    //     Tuple{std::span<real>(u), std::span<real>(v), std::span<real>(w)}};
    auto s = Scalar<std::span<real>>{Tuple{t}, Tuple{u, v, w}};

    constexpr auto tr = vs::transform([](auto&& loc) {
        auto&& [x, y, z] = loc;
        return x * y * z;
    });

    constexpr auto sol = loc.view() | tr;

    s | selector::D = sol;
    REQUIRE(rs::equal(s | selector::D, vs::cartesian_product(x, y, z) | tr));
    REQUIRE(rs::equal(s | selector::Rx, vs::repeat_n(0.0, rx.size())));

    s | selector::Rx = sol;
    REQUIRE(rs::equal(s | selector::Rx, rx | tr));

    s | selector::Ry = sol;
    REQUIRE(rs::equal(s | selector::Ry, ry | tr));

    s | selector::Rz = sol;
    REQUIRE(rs::equal(s | selector::Rz, rz | tr));

    constexpr auto ur = vs::transform([](auto&& loc) {
        auto&& [x, y, z] = loc;
        return x * y * y * z;
    });

    s | selector::R = loc.view() | ur;
    REQUIRE(rs::equal(s | selector::Rx, rx | ur));
    REQUIRE(rs::equal(s | selector::Ry, ry | ur));
    REQUIRE(rs::equal(s | selector::Rz, rz | ur));

    s | selector::R = loc.view() | Tuple{vs::transform(loc_fn<0>{}),
                                         vs::transform(loc_fn<1>{}),
                                         vs::transform(loc_fn<2>{})};

    REQUIRE(rs::equal(u, rx | vs::transform(loc_fn<0>{})));
    REQUIRE(rs::equal(v, ry | vs::transform(loc_fn<1>{})));
    REQUIRE(rs::equal(w, rz | vs::transform(loc_fn<2>{})));
}
