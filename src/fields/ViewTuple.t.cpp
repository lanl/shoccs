#include "ContainerTuple.hpp"
#include "ViewTuple.hpp"

#include <catch2/catch_test_macros.hpp>

#include <range/v3/algorithm/equal.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/repeat_n.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip_with.hpp>

#include <vector>

#include <iostream>

TEST_CASE("construction")
{
    using namespace ccs;
    using namespace ccs::field::tuple;
    using T = std::vector<real>;

    REQUIRE(std::constructible_from<ViewTuple<T&>, T&>);

    auto v = T{1, 2, 3};
    ViewTuple<T&> yy{v};
    ViewTuple<T&> zz{MOVE(yy)};
    ViewTuple<T&> x{zz};

    REQUIRE(rs::equal(x, v));

    auto [y] = x;
    REQUIRE(rs::equal(y, v));

    auto vv = T{3, 4, 5};
    ViewTuple<T&, T&> z{v, vv};
    auto [a, b] = z;

    REQUIRE(rs::equal(a, std::vector<real>{1, 2, 3}));
    REQUIRE(rs::equal(b, std::vector<real>{3, 4, 5}));
    for (auto&& i : b) i *= 2;
    REQUIRE(rs::equal(vv, T{6, 8, 10}));
}

TEST_CASE("AsView")
{
    using namespace ccs;
    using namespace ccs::field::tuple;

    auto a = AsView{vs::iota(0, 10)};
    REQUIRE(rs::size(a) == 10u);
    REQUIRE(rs::equal(a, vs::iota(0, 10)));

    auto b = a;
    REQUIRE(rs::size(a) == 10u);
    REQUIRE(rs::size(b) == rs::size(a));
    REQUIRE(rs::equal(b, vs::iota(0, 10)));

    auto c = MOVE(a);
    REQUIRE(rs::size(c) == 10u);
    REQUIRE(rs::equal(c, vs::iota(0, 10)));
}

TEST_CASE("Assignment with ViewTuple<Vector&>")
{
    using namespace ccs;
    using namespace ccs::field::tuple;
    using T = std::vector<real>;
    using U = ViewTuple<T&>;

    REQUIRE(std::is_assignable_v<U&, real>);
    REQUIRE(std::is_assignable_v<U&, T>);
    REQUIRE(std::is_assignable_v<U&, vs::all_t<T&>>);

    auto v = T{1, 2, 3};

    U x{v};

    x = -1;
    REQUIRE(rs::equal(v, T{-1, -1, -1}));
    REQUIRE(rs::equal(v, x));
    REQUIRE(rs::equal(get<0>(x), v));

    x = vs::iota(1, 10);
    REQUIRE(x.size() == 3u);
    REQUIRE(rs::equal(v, T{1, 2, 3}));

    x = T{-1, -2};
    REQUIRE(x.size() == 3u);
    REQUIRE(rs::equal(v, T{-1, -2, 3}));
}

TEST_CASE("Assignment from Container")
{
    using namespace ccs;
    using namespace ccs::field::tuple;
    using T = std::vector<real>;
    using U = ViewTuple<T&>;

    U x{};

    auto c = ContainerTuple<T>{vs::iota(0, 10)};
    x = c;

    REQUIRE(rs::equal(x, vs::iota(0, 10)));
}

TEST_CASE("Copy with ViewTuple<Vector&>")
{
    using namespace ccs;
    using namespace ccs::field::tuple;
    using T = std::vector<real>;

    auto v = T{1, 2, 3};

    ViewTuple<T&> x{v};

    auto u = T{4, 5, 6};
    ViewTuple<T&> y{u};

    x = y;
    REQUIRE(rs::equal(v, T{4, 5, 6}));

    // auto w = T{3, 4, 5, 6};
    // x = ViewTuple<T&>{w};
    // REQUIRE(rs::equal(w, T{3, 4, 5, 6}));
    // REQUIRE(rs::equal(x, w));
}
#if 0
TEST_CASE("Assignment with ViewTuple<Vector&, Vector&>")
{
    using namespace ccs;
    using namespace ccs::field::tuple;
    using T = std::vector<real>;

    auto u = T{1, 2, 3};
    auto v = T{4, 5, 6, 7};

    ViewTuple<T&, T&> x{u, v};

    x = -1;
    REQUIRE(rs::equal(v, T{-1, -1, -1, -1}));
    REQUIRE(rs::equal(u, T{-1, -1, -1}));
    REQUIRE(rs::equal(v, get<1>(x)));
    REQUIRE(rs::equal(get<0>(x), u));

    {
        auto q = T{6, 7, 8, 9};
        auto r = T{10, 11, 12, 13, 14};
        x = ViewTuple<T&, T&>{q, r};
    }

    REQUIRE(rs::equal(u, T{6, 7, 8}));
    REQUIRE(rs::equal(v, T{10, 11, 12, 13}));
}

TEST_CASE("Assignment with ViewTuple<span>")
{
    using namespace ccs;
    using namespace ccs::field::tuple;
    using T = std::vector<real>;

    auto v = T{1, 2, 3};

    ViewTuple<std::span<real>> x{v};

    x = -1;
    REQUIRE(rs::equal(v, T{-1, -1, -1}));
    REQUIRE(rs::equal(v, x));
    REQUIRE(rs::equal(get<0>(x), v));

    x = vs::iota(1, 10);
    REQUIRE(x.size() == 3u);
    REQUIRE(rs::equal(v, T{1, 2, 3}));

    x = T{-1, -2};
    REQUIRE(x.size() == 3u);
    REQUIRE(rs::equal(v, T{-1, -2, 3}));
}

TEST_CASE("Assignment with ViewTuple of Const")
{
    using namespace ccs::field::tuple;

    constexpr auto f = []<typename... Args>()
    {
        static_assert(!requires(ViewTuple<Args...> t) { t = 1; });
    };
    f.template operator()<const std::vector<int>&>();
    f.template operator()<std::span<const int>>();
    f.template operator()<std::span<const int>&>();
}

TEST_CASE("Copying NonOutput OneTuples")
{
    using namespace ccs;
    using namespace field::tuple;

    using X = decltype(vs::iota(0, 5));
    using T = ViewTuple<X>;

    T t{vs::iota(0, 10)};
    REQUIRE(rs::equal(t, vs::iota(0, 10)));

    {
        T u{};
        u = t;
        REQUIRE(rs::equal(u, vs::iota(0, 10)));
    }

    {
        T u{t};
        REQUIRE(rs::equal(u, vs::iota(0, 10)));
    }

    {
        T u{t};
        T v{MOVE(u)};
        REQUIRE(rs::equal(v, vs::iota(0, 10)));
    }

    {
        T v{};
        T u{t};
        v = MOVE(u);
        REQUIRE(rs::equal(v, vs::iota(0, 10)));
    }

    t = T{vs::iota(5, 10)};
    REQUIRE(rs::equal(t, vs::iota(5, 10)));
}

TEST_CASE("Copying NonOutput TwoTuples")
{
    using namespace ccs;
    using namespace field::tuple;

    using X = decltype(vs::iota(0, 5));
    using T = ViewTuple<X, X>;

    T t{vs::iota(0, 10), vs::iota(5, 10)};
    {
        auto [x, y] = t;
        REQUIRE(rs::equal(x, vs::iota(0, 10)));
        REQUIRE(rs::equal(y, vs::iota(5, 10)));
    }

    {
        T u{};
        u = t;
        auto [x, y] = u;
        REQUIRE(rs::equal(x, vs::iota(0, 10)));
        REQUIRE(rs::equal(y, vs::iota(5, 10)));
    }

    {
        T u{t};
        auto [x, y] = u;
        REQUIRE(rs::equal(x, vs::iota(0, 10)));
        REQUIRE(rs::equal(y, vs::iota(5, 10)));
    }

    {
        T u{t};
        T v{MOVE(u)};
        auto [x, y] = v;
        REQUIRE(rs::equal(x, vs::iota(0, 10)));
        REQUIRE(rs::equal(y, vs::iota(5, 10)));
    }

    {
        T v{};
        T u{t};
        v = MOVE(u);
        auto [x, y] = v;
        REQUIRE(rs::equal(x, vs::iota(0, 10)));
        REQUIRE(rs::equal(y, vs::iota(5, 10)));
    }

    t = T{vs::iota(2, 5), vs::iota(3, 10)};
    {
        auto [x, y] = t;
        REQUIRE(rs::equal(x, vs::iota(2, 5)));
        REQUIRE(rs::equal(y, vs::iota(3, 10)));
    }
}

TEST_CASE("Non-Modifying Math OneTuples")
{
    using namespace ccs;
    using namespace field::tuple;
    using T = std::vector<real>;

    auto u_ = T{1, 2, 3};
    ViewTuple<T&> u{u_};

    static_assert(std::tuple_size_v<std::remove_cvref_t<decltype(u)>> == 1u);

    REQUIRE(rs::equal(1 + u + 1 + u, T{4, 6, 8}));

    auto v_ = T(3);
    ViewTuple<T&> v{v_};

    // conversion
    v = 1 + u + 1 + u;
    REQUIRE(rs::equal(v, T{4, 6, 8}));

    ViewTuple<std::span<real>> w{v_};
    w = 0;
    REQUIRE(rs::equal(v_, T{0, 0, 0}));

    w = 1 + u + 1 + u;
    REQUIRE(rs::equal(w, T{4, 6, 8}));
}

TEST_CASE("Non-Modifying Math TwoTuples")
{
    using namespace ccs;
    using namespace field::tuple;
    using T = std::vector<real>;

    auto a_ = T{1, 2, 3};
    auto b_ = T{3, 4};
    ViewTuple<T&, T&> u{a_, b_};

    static_assert(std::tuple_size_v<std::remove_cvref_t<decltype(u)>> == 2u);

    {
        auto [a, b] = 1 + u + 1 + u;
        REQUIRE(rs::equal(a, T{4, 6, 8}));
        REQUIRE(rs::equal(b, T{8, 10}));
    }

    auto c_ = T(a_.size());
    auto d_ = T(b_.size());
    ViewTuple<std::span<real>, std::span<real>> v{c_, d_};

    v = 1 + u + 1 + u;
    auto [a, b] = v;
    REQUIRE(rs::equal(a, T{4, 6, 8}));
    REQUIRE(rs::equal(b, T{8, 10}));
}

TEST_CASE("Modifying Math OneTuples")
{
    using namespace ccs;
    using namespace field::tuple;
    using T = std::vector<real>;

    auto u_ = T{1, 2, 3};
    ViewTuple<T&> u{u_};

    static_assert(std::tuple_size_v<std::remove_cvref_t<decltype(u)>> == 1u);

    u += 2;
    REQUIRE(rs::equal(u, T{3, 4, 5}));
    // REQUIRE(rs::equal(1 + u + 1 + u, T{4, 6, 8}));

    auto v_ = T{1, 2, 3};
    ViewTuple<T&> v{v_};

    u += v;
    REQUIRE(rs::equal(u, T{4, 6, 8}));
}

TEST_CASE("Pipe Syntax OneTuples")
{
    using namespace ccs;
    using namespace field::tuple;
    using T = std::vector<int>;

    auto a = T{1, 2, 3};
    auto u = ViewTuple{a};

    REQUIRE(rs::equal(u | vs::transform([](auto&& i) { return i * i; }), T{1, 4, 9}));

    auto b = T(3);
    auto v = ViewTuple{b};

    v = u | vs::transform([](auto&& i) { return i * i; });
    REQUIRE(rs::equal(b, T{1, 4, 9}));
}

TEST_CASE("Pipe Syntax TwoTuples")
{
    using namespace ccs;
    using namespace field::tuple;

    const auto i = vs::iota(0, 10);
    const auto j = vs::iota(-10, 10);
    constexpr auto f = [](auto&& i) { return i + i; };

    auto v = ViewTuple{vs::iota(0, 10), vs::iota(-10, 10)};

    auto u = v | vs::transform(f);

    auto [a, b] = u;

    REQUIRE(rs::equal(a, vs::zip_with(std::plus{}, i, i)));
    REQUIRE(rs::equal(b, vs::zip_with(std::plus{}, j, j)));
}

TEST_CASE("MultiPipe Syntax")
{
    using namespace ccs;
    using namespace field::tuple;

    const auto i = vs::iota(0, 10);
    const auto j = vs::iota(-10, 10);

    constexpr auto f = [](auto&& i) { return i + i; };
    constexpr auto g = [](auto&& i) { return i * i; };

    auto v = ViewTuple{vs::iota(0, 10), vs::iota(-10, 10)};

    auto u = v | std::tuple{vs::transform(f), vs::transform(g)};

    auto [a, b] = u;

    REQUIRE(rs::equal(a, vs::zip_with(std::plus{}, i, i)));
    REQUIRE(rs::equal(b, vs::zip_with(std::multiplies{}, j, j)));
}
#endif