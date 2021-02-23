#include "types.hpp"

#include "Scalar.hpp"
#include "Vector.hpp"

#include "Selector.hpp"

#include "lift.hpp"
#include "views.hpp"

#include <catch2/catch_test_macros.hpp>
#include <range/v3/algorithm/equal.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/transform.hpp>

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
    auto s = SimpleVector<T>{};
}

TEST_CASE("selection")
{
    using namespace ccs;
    using namespace field::tuple;

    // get non-const versions
    auto x = global::x;
    auto y = global::y;
    auto z = global::z;
    auto rx = global::rx | vs::transform([](auto&& i) { return i[0]; }) |
              rs::to<std::vector<real>>();
    auto ry = global::ry | vs::transform([](auto&& i) { return i[1]; }) |
              rs::to<std::vector<real>>();
    auto rz = global::rz | vs::transform([](auto&& i) { return i[2]; }) |
              rs::to<std::vector<real>>();

    auto s = field::Vector(&global::loc, Tuple{x, y, z}, Tuple{rx, ry, rz});

    // Add tests to verify the type of the containers and views.
    // THey may point to scalar needing to be a tuple of nested tuples
    REQUIRE(rs::equal(x, s | selector::Dx));
    REQUIRE(rs::equal(y, s | selector::Dy));
    REQUIRE(rs::equal(z, s | selector::Dz));
    REQUIRE(rs::equal(rx, s | selector::Rx));
    REQUIRE(rs::equal(ry, s | selector::Ry));
    REQUIRE(rs::equal(rz, s | selector::Rz));

    // modify selection
    s | selector::D = 0;
    REQUIRE(rs::equal(x, vs::repeat_n(0, global::x.size())));
    REQUIRE(rs::equal(y, vs::repeat_n(0, global::y.size())));
    REQUIRE(rs::equal(z, vs::repeat_n(0, global::z.size())));

    s | selector::Rz = -1;
    REQUIRE(rs::equal(rz, vs::repeat_n(-1, global::rz.size())));

    s | selector::Rxyz = 2;
    REQUIRE(rs::equal(rx, vs::repeat_n(2, global::rx.size())));
    REQUIRE(rs::equal(ry, vs::repeat_n(2, global::ry.size())));
    REQUIRE(rs::equal(rz, vs::repeat_n(2, global::rz.size())));
}

TEST_CASE("math")
{
    using namespace ccs;
    using namespace ccs::field;
    auto s = SimpleVector<std::vector<int>>{
        &global::loc,
        Tuple{vs::iota(1, 5), vs::iota(2, 7), vs::iota(1, 3)},
        Tuple{vs::iota(6, 10), vs::iota(6, 12), vs::iota(10, 15)}};

    auto q = s + 1;
    REQUIRE(rs::equal(vs::iota(2, 6), q | selector::Dx));
    REQUIRE(rs::equal(vs::iota(3, 8), q | selector::Dy));
    REQUIRE(rs::equal(vs::iota(2, 4), q | selector::Dz));
    REQUIRE(rs::equal(vs::iota(7, 11), q | selector::Rx));
    REQUIRE(rs::equal(vs::iota(7, 13), q | selector::Ry));
    REQUIRE(rs::equal(vs::iota(11, 16), q | selector::Rz));

    auto r = q + s;
    constexpr auto plus = [](auto&& a, auto&& b) {
        return vs::zip_with(std::plus{}, FWD(a), FWD(b));
    };
    REQUIRE(rs::equal(plus(vs::iota(2, 6), vs::iota(1, 5)), r | selector::Dx));
    REQUIRE(rs::equal(plus(vs::iota(11, 16), vs::iota(10, 15)), r | selector::Rz));

    SimpleVector<std::vector<int>> t{r};
    REQUIRE(rs::equal(t | selector::Dz, r | selector::Dz));
    REQUIRE(rs::equal(t | selector::Ry, r | selector::Ry));
}

TEST_CASE("lifting single arg")
{
    using namespace ccs;
    using namespace ccs::field;
    auto s = SimpleVector<std::vector<int>>{
        &global::loc,
        Tuple{vs::iota(1, 5), vs::iota(2, 7), vs::iota(1, 3)},
        Tuple{vs::iota(6, 10), vs::iota(6, 12), vs::iota(10, 15)}};

    constexpr auto f = lift([](auto&& x) { return std::abs(x) + 1; });

    auto j = f(s);
    auto k = s + 1;

    REQUIRE(rs::equal(j | selector::Dx, k | selector::Dx));
    REQUIRE(rs::equal(j | selector::Dy, k | selector::Dy));
    REQUIRE(rs::equal(j | selector::Dz, k | selector::Dz));
    REQUIRE(rs::equal(j | selector::Rx, k | selector::Rx));
    REQUIRE(rs::equal(j | selector::Ry, k | selector::Ry));
    REQUIRE(rs::equal(j | selector::Rz, k | selector::Rz));
}

TEST_CASE("lifting multiple args")
{
    using namespace ccs;
    using namespace ccs::field;
    auto x = SimpleVector<std::vector<int>>{
        &global::loc,
        Tuple{vs::iota(1, 5), vs::iota(2, 7), vs::iota(1, 3)},
        Tuple{vs::iota(6, 10), vs::iota(6, 12), std::vector{-4, 5, -10, 6}}};
    auto y = SimpleVector<std::vector<int>>{
        &global::loc,
        Tuple{vs::iota(2, 6), vs::iota(1, 6), vs::iota(2, 4)},
        Tuple{vs::iota(5, 9), vs::iota(-6, 0), std::vector{10, -8, 2, -7}}};

    constexpr auto f =
        lift([](auto&& x, auto&& y) { return std::max(std::abs(x), std::abs(y)); });

    auto z = f(x, y);

    REQUIRE(rs::equal(z | selector::Dx, vs::iota(2, 6)));
    REQUIRE(rs::equal(z | selector::Dy, vs::iota(2, 7)));
    REQUIRE(rs::equal(z | selector::Dz, vs::iota(2, 4)));
    REQUIRE(rs::equal(z | selector::Rx, vs::iota(6, 10)));
    REQUIRE(rs::equal(z | selector::Ry, vs::iota(6, 12)));
    REQUIRE(rs::equal(z | selector::Rz, std::vector{10, 8, 10, 7}));
}

TEST_CASE("mesh location")
{
    using namespace ccs;
    using namespace ccs::field;
    using namespace global;

    const auto sz = x.size() * y.size() * z.size();

    auto s = SimpleVector<std::vector<real>>{
        &loc, Tuple{sz, sz, sz}, Tuple{rx.size(), ry.size(), rz.size()}};

    constexpr auto t = vs::transform([](auto&& loc) {
        auto&& [x, y, z] = loc;
        return x * y * z;
    });

    constexpr auto sol = mesh::location | t;

    s | selector::D = sol;
    REQUIRE(rs::equal(s | selector::Dx, vs::cartesian_product(x, y, z) | t));
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
