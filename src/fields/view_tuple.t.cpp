#include "view_tuple.hpp"
#include "container_tuple.hpp"

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

using namespace ccs;

TEST_CASE("construction")
{
    using T = std::vector<real>;

    REQUIRE(std::constructible_from<view_tuple<T&>, T&>);

    auto v = T{1, 2, 3};
    view_tuple<T&> yy{v};
    view_tuple<T&> zz{MOVE(yy)};
    view_tuple<T&> x{zz};

    REQUIRE(x == v);

    auto [y] = x;
    REQUIRE(rs::equal(y, v));

    auto vv = T{3, 4, 5};
    view_tuple<T&, T&> z{v, vv};
    auto [a, b] = z;

    REQUIRE(rs::equal(a, std::vector<real>{1, 2, 3}));
    REQUIRE(rs::equal(b, std::vector<real>{3, 4, 5}));
    for (auto&& i : b) i *= 2;
    REQUIRE(vv == T{6, 8, 10});
}

TEST_CASE("single_view")
{
    auto a = single_view{vs::iota(0, 10)};
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

TEST_CASE("Assignment with view_tuple<Vector&>")
{
    using T = std::vector<real>;
    using U = view_tuple<T&>;

    REQUIRE(std::is_assignable_v<U&, real>);
    REQUIRE(std::is_assignable_v<U&, T>);
    REQUIRE(std::is_assignable_v<U&, vs::all_t<T&>>);

    auto v = T{1, 2, 3};

    U x{v};

    x = -1;
    REQUIRE(v == T{-1, -1, -1});
    REQUIRE(x == v);
    REQUIRE(rs::equal(get<0>(x), v));

    x = vs::iota(1, 10);
    REQUIRE(x.size() == 9u);
    REQUIRE(rs::equal(v, vs::iota(1, 10)));

    x = T{-1, -2};
    REQUIRE(x.size() == 2u);
    REQUIRE(v == T{-1, -2});
}

TEST_CASE("Assignment from Container")
{
    using T = std::vector<real>;
    using U = view_tuple<T&>;

    U x{};

    auto c = container_tuple<T>{vs::iota(0, 10)};
    x = c;

    REQUIRE(x == vs::iota(0, 10));
}

TEST_CASE("Copy with view_tuple<Vector&>")
{
    using T = std::vector<real>;

    auto v = T{1, 2, 3};

    view_tuple<T&> x{v};

    auto u = T{4, 5, 6};
    view_tuple<T&> y{u};

    x = y;
    REQUIRE(v == T{4, 5, 6});

    auto w = T{3, 4, 5, 6};
    x = view_tuple<T&>{w};
    REQUIRE(w == T{3, 4, 5, 6});
    REQUIRE(x == w);
}

TEST_CASE("Assignment with view_tuple<Vector&, Vector&>")
{
    using T = std::vector<real>;

    auto u = T{1, 2, 3};
    auto v = T{4, 5, 6, 7};

    view_tuple<T&, T&> x{u, v};

    x = -1;
    REQUIRE(x == view_tuple{vs::repeat_n(-1, u.size()), vs::repeat_n(-1, v.size())});
    REQUIRE(rs::equal(v, get<1>(x)));
    REQUIRE(rs::equal(get<0>(x), u));

    auto q = T{6, 7, 8, 9};
    auto r = T{10, 11, 12, 13, 14};
    x = view_tuple<T&, T&>{q, r};

    REQUIRE(view_tuple{u, v} == view_tuple{q, r});
}

TEST_CASE("Assignment with view_tuple<span>")
{
    using T = std::vector<real>;

    auto v = T{1, 2, 3};

    view_tuple<std::span<real>> x{v};

    x = -1;
    REQUIRE(v == T{-1, -1, -1});
    REQUIRE(x == v);
    REQUIRE(rs::equal(get<0>(x), v));

    x = vs::iota(1, 10);
    REQUIRE(x.size() == 3u);
    REQUIRE(v == T{1, 2, 3});

    x = T{-1, -2};
    REQUIRE(x.size() == 3u);
    REQUIRE(v == T{-1, -2, 3});
}

TEST_CASE("Assignment with view_tuple of Const")
{
    constexpr auto f = []<typename... Args>()
    {
        static_assert(!requires(view_tuple<Args...> t) { t = 1; });
    };
    f.template operator()<const std::vector<int>&>();
    f.template operator()<std::span<const int>>();
    f.template operator()<std::span<const int>&>();
}

TEST_CASE("Copying NonOutput OneTuples")
{
    using X = decltype(vs::iota(0, 5));
    using T = view_tuple<X>;

    T t{vs::iota(0, 10)};
    REQUIRE(t == vs::iota(0, 10));

    {
        T u{};
        u = t;
        REQUIRE(u == vs::iota(0, 10));
    }

    {
        T u{t};
        REQUIRE(u == vs::iota(0, 10));
    }

    {
        T u{t};
        T v{MOVE(u)};
        REQUIRE(v == vs::iota(0, 10));
    }

    {
        T v{};
        T u{t};
        v = MOVE(u);
        REQUIRE(v == vs::iota(0, 10));
    }

    t = T{vs::iota(5, 10)};
    REQUIRE(t == vs::iota(5, 10));
}

TEST_CASE("Copying NonOutput TwoTuples")
{
    using X = decltype(vs::iota(0, 5));
    using T = view_tuple<X, X>;

    T t{vs::iota(0, 10), vs::iota(5, 10)};
    {
        auto [x, y] = t;
        REQUIRE(rs::equal(x, vs::iota(0, 10)));
        REQUIRE(rs::equal(y, vs::iota(5, 10)));
    }

    {
        T u{};
        u = t;
        REQUIRE(t == u);
    }

    {
        T u{t};
        REQUIRE(t == u);
    }

    {
        T u{t};
        T v{MOVE(u)};
        REQUIRE(t == u);
    }

    {
        T v{};
        T u{t};
        v = MOVE(u);
        REQUIRE(t == v);
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
    using T = std::vector<real>;

    auto u_ = T{1, 2, 3};
    view_tuple<T&> u{u_};

    static_assert(std::tuple_size_v<std::remove_cvref_t<decltype(u)>> == 1u);

    REQUIRE(1 + u + 1 + u == T{4, 6, 8});

    auto v_ = T(3);
    view_tuple<T&> v{v_};

    // conversion
    v = 1 + u + 1 + u;
    REQUIRE(v == T{4, 6, 8});

    view_tuple<std::span<real>> w{v_};
    w = 0;
    REQUIRE(v_ == T{0, 0, 0});

    w = 1 + u + 1 + u;
    REQUIRE(w == T{4, 6, 8});
}

TEST_CASE("Non-Modifying Math TwoTuples")
{
    using T = std::vector<real>;
    using C = container_tuple<T, T>;

    auto a_ = T{1, 2, 3};
    auto b_ = T{3, 4};
    view_tuple<T&, T&> u{a_, b_};

    static_assert(std::tuple_size_v<std::remove_cvref_t<decltype(u)>> == 2u);

    {
        auto [a, b] = 1 + u + 1 + u;
        REQUIRE(rs::equal(a, T{4, 6, 8}));
        REQUIRE(rs::equal(b, T{8, 10}));
    }

    auto c_ = T(a_.size());
    auto d_ = T(b_.size());
    view_tuple<std::span<real>, std::span<real>> v{c_, d_};

    v = 1 + u + 1 + u;
    REQUIRE(v == C{T{4, 6, 8}, T{8, 10}});
}

TEST_CASE("Modifying Math OneTuples")
{
    using T = std::vector<real>;

    auto u_ = T{1, 2, 3};
    view_tuple<T&> u{u_};

    static_assert(std::tuple_size_v<std::remove_cvref_t<decltype(u)>> == 1u);

    u += 2;
    REQUIRE(u == T{3, 4, 5});

    auto v_ = T{1, 2, 3};
    view_tuple<T&> v{v_};

    u += v;
    REQUIRE(u == T{4, 6, 8});
}

TEST_CASE("Pipe Syntax OneTuples")
{
    using T = std::vector<int>;

    auto a = T{1, 2, 3};
    auto u = view_tuple{a};

    REQUIRE(rs::equal(u | vs::transform([](auto&& i) { return i * i; }), T{1, 4, 9}));

    auto b = T(3);
    auto v = view_tuple{b};

    v = u | vs::transform([](auto&& i) { return i * i; });
    REQUIRE(b == T{1, 4, 9});
}

TEST_CASE("Pipe Syntax TwoTuples")
{
    const auto i = vs::iota(0, 10);
    const auto j = vs::iota(-10, 10);
    constexpr auto f = [](auto&& i) { return i + i; };

    auto v = view_tuple{vs::iota(0, 10), vs::iota(-10, 10)};

    REQUIRE((v | vs::transform(f)) ==
            view_tuple{vs::zip_with(std::plus{}, i, i), vs::zip_with(std::plus{}, j, j)});
}

TEST_CASE("MultiPipe Syntax")
{
    const auto i = vs::iota(0, 10);
    const auto j = vs::iota(-10, 10);

    constexpr auto f = [](auto&& i) { return i + i; };
    constexpr auto g = [](auto&& i) { return i * i; };

    auto v = view_tuple{vs::iota(0, 10), vs::iota(-10, 10)};

    REQUIRE((v | std::tuple{vs::transform(f), vs::transform(g)}) ==
            view_tuple{vs::zip_with(std::plus{}, i, i),
                       vs::zip_with(std::multiplies{}, j, j)});
}

// type for mocking selection
template <typename R>
struct X : R {

    X() = default;
    constexpr X(R r) : R{MOVE(r)} {}

    template <ViewClosures F>
    X& operator=(F f)
    {
        auto rng = *this | f;
        R::operator=(rng);
        return *this;
    }
};

namespace ranges
{
template <ccs::All T>
inline constexpr bool enable_view<X<T>> = true;
}

TEST_CASE("pass through assignment - vector")
{
    using T = std::vector<int>;
    using V = view_tuple<T&>;

    constexpr auto closure =
        rs::make_view_closure([](auto&&) { return vs::iota(0, 10); });

    auto t = T{};
    auto a = V{t};
    static_assert(is_ref_view<decltype(get<0>(a))>::value);
    a = rs::empty_view<int>{} | closure | vs::transform([](auto&& i) { return i + 1; });
    REQUIRE(rs::equal(t, vs::iota(1, 11)));

    {
        t.clear();
        auto x = X{V{t}};
        x = closure | vs::transform([](auto&& i) { return i - 1; });
        REQUIRE(rs::equal(t, vs::iota(-1, 9)));
    }

    {
        t.clear();
        static_assert(All<X<V>>);
        auto x = view_tuple{X{V{t}}};
        static_assert(AssignableDirect<view_tuple_base<X<V>>, decltype(closure)>);
        static_assert(AssignableDirect<view_tuple_base<X<V>>&, decltype(closure)>);
        x = closure | vs::transform([](auto&& i) { return i + 1; });
        REQUIRE(rs::equal(t, vs::iota(1, 11)));
    }
}

TEST_CASE("pass through assignment - span")
{
    using T = std::span<int>;
    using V = view_tuple<T>;

    static_assert(All<T>);
    static_assert(All<T&>);
    static_assert(All<V>);
    static_assert(All<view_tuple<std::span<int>&>>);

    constexpr auto closure =
        rs::make_view_closure([](auto&&) { return vs::iota(0, 10); });

    auto t = std::vector<int>(10);
    auto a = V{t};
    a = rs::empty_view<int>{} | closure | vs::transform([](auto&& i) { return i + 1; });
    REQUIRE(rs::equal(t, vs::iota(1, 11)));

    {
        auto x = X{V{t}};
        x = closure | vs::transform([](auto&& i) { return i - 1; });
        REQUIRE(rs::equal(t, vs::iota(-1, 9)));
    }

    {
        static_assert(All<X<V>>);
        auto x = view_tuple{X{V{t}}};
        static_assert(AssignableDirect<view_tuple_base<X<V>>, decltype(closure)>);
        static_assert(AssignableDirect<view_tuple_base<X<V>>&, decltype(closure)>);
        x = closure | vs::transform([](auto&& i) { return i + 1; });
        REQUIRE(rs::equal(t, vs::iota(1, 11)));
    }
}
