#include "types.hpp"

#include "Scalar.hpp"

#include "Selector.hpp"

#include "lift.hpp"
#include "views.hpp"

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
const auto loc = ccs::mesh::Location{x, y, z, rx, ry, rz};

} // namespace global

TEST_CASE("construction")
{
    using namespace ccs::field;
    using T = std::vector<int>;
    auto s = SimpleScalar<T>{};
}
TEST_CASE("selection")
{
    using namespace ccs;
    using namespace field::tuple;
    // some initialization
    auto v = std::vector{1, 2};
    auto rx = std::vector{3, 4};
    auto ry = std::vector{5, 6};
    auto rz = std::vector{7, 8};
    // auto s = field::Scalar<std::vector<int>&>{tag{}, Tuple{v}, Tuple{rx, ry, rz}};
    auto s = field::Scalar(&global::loc, Tuple{v}, Tuple{rx, ry, rz});

    static_assert(std::same_as<decltype(s.get<0>()), Tuple<std::vector<int>&>&>);
    static_assert(std::same_as<decltype(view<0>(s.get<0>())), decltype(vs::all(v))>);

    // Add tests to verify the type of the containers and views.
    // THey may point to scalar needing to be a tuple of nested tuples
    REQUIRE(rs::equal(v, s | selector::D));
    REQUIRE(rs::equal(rx, s | selector::Rx));
    REQUIRE(rs::equal(ry, s | selector::Ry));
    REQUIRE(rs::equal(rz, s | selector::Rz));

    // static_assert(field::tuple::All<std::remove_cvref_t<decltype(s | selector::D)>>);
    using T = decltype(s | selector::D);
    static_assert(All<ranges::ref_view<std::vector<int>>>);
    static_assert(rs::range<T>);
    static_assert(rs::semiregular<T>);
    static_assert(rs::enable_view<T>);
    static_assert(field::tuple::All<T>);

    REQUIRE(rs::equal(s | selector::D | vs::transform([](auto&& x) { return x * x * x; }),
                      std::vector{1, 8}));

    // modify selection
    s | selector::D = 0;
    REQUIRE(rs::equal(v, vs::repeat_n(0, 2)));

    s | selector::Rz = -1;
    REQUIRE(rs::equal(rz, vs::repeat_n(-1, 2)));

    s | selector::Rxyz = 2;
    REQUIRE(rs::equal(v, vs::repeat_n(0, 2)));
    REQUIRE(rs::equal(rx, vs::repeat_n(2, 2)));
}

TEST_CASE("math")
{
    using namespace ccs;
    using namespace ccs::field;
    auto s = SimpleScalar<std::vector<int>>{
        &global::loc,
        Tuple{vs::iota(1, 5)},
        Tuple{vs::iota(6, 10), vs::iota(6, 12), vs::iota(10, 15)}};

    auto q = s + 1;
    REQUIRE(rs::equal(vs::iota(2, 6), q | selector::D));
    REQUIRE(rs::equal(vs::iota(7, 11), q | selector::Rx));
    REQUIRE(rs::equal(vs::iota(7, 13), q | selector::Ry));
    REQUIRE(rs::equal(vs::iota(11, 16), q | selector::Rz));
    auto r = q + s;
    constexpr auto plus = [](auto&& a, auto&& b) {
        return vs::zip_with(std::plus{}, FWD(a), FWD(b));
    };
    REQUIRE(rs::equal(plus(vs::iota(2, 6), vs::iota(1, 5)), r | selector::D));
    REQUIRE(rs::equal(plus(vs::iota(11, 16), vs::iota(10, 15)), r | selector::Rz));

    SimpleScalar<std::vector<int>> t = r;
    REQUIRE(rs::equal(t | selector::D, r | selector::D));
    REQUIRE(rs::equal(t | selector::Ry, r | selector::Ry));
}

TEST_CASE("lifting single arg")
{
    using namespace ccs;
    using namespace ccs::field;
    auto s = SimpleScalar<std::vector<int>>{
        &global::loc,
        Tuple{vs::iota(1, 5)},
        Tuple{vs::iota(6, 10), vs::iota(6, 12), vs::iota(10, 15)}};

    constexpr auto f = lift([](auto&& x) { return std::abs(x) + 1; });

    auto j = f(s);
    auto k = s + 1;

    REQUIRE(rs::equal(j | selector::D, k | selector::D));
    REQUIRE(rs::equal(j | selector::Rx, k | selector::Rx));
    REQUIRE(rs::equal(j | selector::Ry, k | selector::Ry));
    REQUIRE(rs::equal(j | selector::Rz, k | selector::Rz));
}

TEST_CASE("lifting multiple args")
{
    using namespace ccs;
    using namespace ccs::field;
    auto x = SimpleScalar<std::vector<int>>{
        &global::loc,
        Tuple{vs::iota(1, 5)},
        Tuple{vs::iota(6, 10), vs::iota(6, 12), std::vector{-4, 5, -10, 6}}};
    auto y = SimpleScalar<std::vector<int>>{
        &global::loc,
        Tuple{vs::iota(2, 6)},
        Tuple{vs::iota(5, 9), vs::iota(-6, 0), std::vector{10, -8, 2, -7}}};

    constexpr auto f =
        lift([](auto&& x, auto&& y) { return std::max(std::abs(x), std::abs(y)); });

    auto z = f(x, y);

    REQUIRE(rs::equal(z | selector::D, vs::iota(2, 6)));
    REQUIRE(rs::equal(z | selector::Rx, vs::iota(6, 10)));
    REQUIRE(rs::equal(z | selector::Ry, vs::iota(6, 12)));
    REQUIRE(rs::equal(z | selector::Rz, std::vector{10, 8, 10, 7}));
}

TEST_CASE("mesh location vector")
{
    using namespace ccs;
    using namespace ccs::field;
    using namespace global;

    auto s = SimpleScalar<std::vector<real>>{&loc,
                                             Tuple{x.size() * y.size() * z.size()},
                                             Tuple{rx.size(), ry.size(), rz.size()}};

    constexpr auto t = vs::transform([](auto&& loc) {
        auto&& [x, y, z] = loc;
        return x * y * z;
    });

    constexpr auto sol = mesh::location | t;

    s | selector::D = sol;
    REQUIRE(rs::equal(s | selector::D, vs::cartesian_product(x, y, z) | t));
    REQUIRE(rs::equal(s | selector::Rx, vs::repeat_n(0.0, rx.size())));

    s | selector::Rx = sol;
    REQUIRE(rs::equal(s | selector::Rx, rx | t));

    s | selector::Ry = sol;
    REQUIRE(rs::equal(s | selector::Ry, ry | t));

    s | selector::Rz = sol;
    REQUIRE(rs::equal(s | selector::Rz, rz | t));

    constexpr auto u = vs::transform([](auto&& loc) {
        auto&& [x, y, z] = loc;
        return x * y * y * z;
    });

    s | selector::Rxyz = mesh::location | u;
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

    auto s = SimpleScalar<std::span<real>>{
        &loc,
        Tuple{std::span<real>(t)},
        Tuple{std::span<real>(u), std::span<real>(v), std::span<real>(w)}};

    constexpr auto tr = vs::transform([](auto&& loc) {
        auto&& [x, y, z] = loc;
        return x * y * z;
    });

    constexpr auto sol = mesh::location | tr;

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

    s | selector::Rxyz = mesh::location | ur;
    REQUIRE(rs::equal(s | selector::Rx, rx | ur));
    REQUIRE(rs::equal(s | selector::Ry, ry | ur));
    REQUIRE(rs::equal(s | selector::Rz, rz | ur));

    s | selector::Rxyz = mesh::location | Tuple{vs::transform(loc_fn<0>{}),
                                                vs::transform(loc_fn<1>{}),
                                                vs::transform(loc_fn<2>{})};

    REQUIRE(rs::equal(u, rx | vs::transform(loc_fn<0>{})));
    REQUIRE(rs::equal(v, ry | vs::transform(loc_fn<1>{})));
    REQUIRE(rs::equal(w, rz | vs::transform(loc_fn<2>{})));
}
