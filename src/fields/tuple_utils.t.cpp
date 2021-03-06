#include "tuple_utils.hpp"

#include <catch2/catch_test_macros.hpp>

#include <range/v3/all.hpp>

#include <iostream>
#include <vector>

using namespace ccs;

TEST_CASE("test")
{

    using X = std::tuple<int>;
    using Y = std::tuple<float, long>;
    using A = std::tuple<X, Y>;
    using XX = std::tuple<double>;
    using YY = std::tuple<char, unsigned>;
    using B = std::tuple<XX, YY>;
    using F = decltype([](auto&&...) {});

    using C = is_nested_invocable<F, A&, B&>;
    static_assert(std::same_as<C,
                               mp_list<std::is_invocable<F, int&, double&>,
                                       std::is_invocable<F, float&, char&>,
                                       std::is_invocable<F, long&, unsigned&>>>);
    static_assert(mp_apply<mp_all, C>::value);
    static_assert(NestedInvocableOver<F, A&, B&>);
    // static_assert(std::is_same<C, mp_list<t::is_nested_invocable<F, int&, double&>>>);
    // using C = mp_list<tuple_get_types<A&>, tuple_get_types<B&>>;
    // static_assert(std::same_as<C, mp_list<mp_list<X&, Y&>, mp_list<XX&, YY&>>>);

    // using D = mp_transform_q<mp_bind_front<t::is_nested_invocable, F>,
    //                          tuple_get_types<A&>,
    //                          tuple_get_types<B&>>;
    // static_assert(std::same_as<D,
    //                            mp_list<t::is_nested_invocable<F, X&, XX&>,
    //                                    t::is_nested_invocable<F, Y&, YY&>>>);
    // using E = mp_transform<mp_list, tuple_get_types<A&>, tuple_get_types<B&>>;
    // static_assert(std::same_as<E, mp_list<mp_list<X&, XX&>, mp_list<Y&, YY&>>>);

    // using G = mp_transform_q<mp_bind_front<t::is_nested_invocable, F>,
    //                          tuple_get_types<X&>,
    //                          tuple_get_types<XX&>>;
    // static_assert(std::same_as<G, mp_list<t::is_nested_invocable<F, int&, double&>>>);

    // need to transform <F, A> into
    // invocable<F, int>
    // invocable<F, float>
    // invocable<F, char>
    // using B = mp_transform_q<mp_bind_front<t::is_pipeable, T>, A>;
    // static_assert(std::same_as<B, mp_list<t::is_pipeable<T, V>>>);
    // using B = mp_list<void, int&, const std::vector<int>&>;

    // using C = mp_transform<mp_list, A, B>;
    // static_assert(std::same_as<C,
    //                            mp_list<mp_list<char*, void>,
    //                                    mp_list<const double&, int&>,
    //                                    mp_list<float&, const std::vector<int>&>>>);
    // using D = std::tuple<int, int&, const int&>;
    // using E = tuple_get_types<D>;
    // static_assert(std::same_as<E, mp_list<int&&, int&, const int&>>);

    // using F = tuple_get_types<D&>;
    // static_assert(std::same_as<F, mp_list<int&, int&, const int&>>);

    // using G = tuple_get_types<const D&>;
    // static_assert(std::same_as<G, mp_list<const int&, int&, const int&>>);
}

TEST_CASE("for_each")
{
    using T = std::vector<int>;

    auto s = std::tuple{T{1, 2, 3}, T{4, 5, 6}, T{7, 8, 9}};
    auto t = std::tuple{T{4, 5, 6}, T{1, 2, 3}, T{-1, -2, -3}};

    for_each(
        [](T& v) {
            for (auto&& i : v) i += 1;
        },
        s);
    REQUIRE(rs::equal(get<0>(s), T{2, 3, 4}));
    REQUIRE(rs::equal(get<1>(s), T{5, 6, 7}));
    REQUIRE(rs::equal(get<2>(s), T{8, 9, 10}));

    for_each(
        [](auto& x, auto& y) {
            using std::swap;
            swap(x, y);
        },
        s,
        t);
    REQUIRE(rs::equal(get<0>(t), T{2, 3, 4}));
    REQUIRE(rs::equal(get<1>(t), T{5, 6, 7}));
    REQUIRE(rs::equal(get<2>(t), T{8, 9, 10}));

    REQUIRE(rs::equal(get<0>(s), T{4, 5, 6}));
    REQUIRE(rs::equal(get<1>(s), T{1, 2, 3}));
    REQUIRE(rs::equal(get<2>(s), T{-1, -2, -3}));

    for_each(
        []<auto I>(mp_size_t<I>, auto& v) {
            for (auto&& i : v) i += I;
        },
        s);
    REQUIRE(rs::equal(get<0>(s), T{4, 5, 6}));
    REQUIRE(rs::equal(get<1>(s), T{2, 3, 4}));
    REQUIRE(rs::equal(get<2>(s), T{1, 0, -1}));

    for_each(std::tuple{[](auto&& x, auto&& y) {
                            using std::swap;
                            swap(x, y);
                        },
                        [](auto& x, auto& y) {
                            for (auto&& [i, j] : vs::zip(x, y)) {
                                i += 1;
                                j -= 1;
                            }
                        },
                        [](auto&& x, auto&& y) {
                            for (auto&& [i, j] : vs::zip(x, y)) {
                                i *= 2;
                                j *= 3;
                            }
                        }},
             s,
             t);
    // swapped
    REQUIRE(rs::equal(get<0>(s), T{2, 3, 4}));
    REQUIRE(rs::equal(get<0>(t), T{4, 5, 6}));

    REQUIRE(rs::equal(get<1>(s), T{3, 4, 5}));
    REQUIRE(rs::equal(get<1>(t), T{4, 5, 6}));

    REQUIRE(rs::equal(get<2>(s), T{2, 0, -2}));
    REQUIRE(rs::equal(get<2>(t), T{24, 27, 30}));
}

TEST_CASE("nested for_each")
{

    using T = std::vector<int>;

    auto s = std::tuple{std::tuple{T{1, 2}}, std::tuple{T{3, 4}, T{5}}};

    for_each(
        [](T& v) {
            for (auto&& i : v) i += 1;
        },
        s);
    REQUIRE(get<0>(get<0>(s)) == T{2, 3});
    REQUIRE(get<0>(get<1>(s)) == T{4, 5});
    REQUIRE(get<1>(get<1>(s)) == T{6});

    auto t = std::tuple{std::tuple{T{1, 2}}, std::tuple{T{3, 4}, T{5}}};

    for_each(
        [](const T& u, T& v) {
            for (auto&& [i, j] : vs::zip(u, v)) j *= i;
        },
        s,
        t);
    REQUIRE(get<0>(get<0>(t)) == T{2, 6});
    REQUIRE(get<0>(get<1>(t)) == T{12, 20});
    REQUIRE(get<1>(get<1>(t)) == T{30});
}

TEST_CASE("reduce")
{
    auto a = reduce([](auto&& acc, auto&& item) { return std::max(FWD(acc), FWD(item)); },
                    std::tuple{std::tuple{2}, std::tuple{4}, std::tuple{1}},
                    -1234);
    REQUIRE(a == 4);
}

TEST_CASE("transform")
{

    using T = std::vector<int>;

    auto s = std::tuple{T{1, 2, 3}, T{4, 5, 6, 7, 8, 9}, T{4, 5}};
    auto t = std::tuple(T{4, 5, 6}, T{1, 2, 3, 4, 5, 6}, T{6, 7});

    {
        auto r = transform([](auto&& vec) { return rs::accumulate(FWD(vec), 0); }, s);

        auto&& [x, y, z] = r;
        REQUIRE(x == 6);
        REQUIRE(y == 39);
        REQUIRE(z == 9);
    }

    {
        auto r = transform(
            [](auto&&... vec) { return (rs::accumulate(FWD(vec), 0) + ...); }, s, t);

        auto&& [x, y, z] = r;
        REQUIRE(x == 21);
        REQUIRE(y == 60);
        REQUIRE(z == 22);
    }

    {
        constexpr auto f = []<auto I>(mp_size_t<I>, auto&& vec)
        {
            return rs::accumulate(FWD(vec), I);
        };
        auto r = transform(f, s);

        auto&& [x, y, z] = r;
        REQUIRE(x == 6);
        REQUIRE(y == 40);
        REQUIRE(z == 11);
    }

    {
        constexpr auto f = []<auto I>(mp_size_t<I>, auto&&... vec)
        {
            return (rs::accumulate(FWD(vec), I) + ...);
        };
        auto r = transform(f, s, t);

        auto&& [x, y, z] = r;
        REQUIRE(x == 21);
        REQUIRE(y == 62);
        REQUIRE(z == 26);
    }

    {
        constexpr auto f = [](auto&& vec) { return rs::accumulate(FWD(vec), -6); };
        constexpr auto g = [](auto&& vec) { return rs::accumulate(FWD(vec), -39); };
        constexpr auto h = [](auto&& vec) { return rs::accumulate(FWD(vec), -9); };

        auto r = transform(std::tuple{f, g, h}, s);

        auto&& [x, y, z] = r;
        REQUIRE(x == 0);
        REQUIRE(y == 0);
        REQUIRE(z == 0);
    }

    {
        constexpr auto f = [](auto&& x, auto&& y) {
            return rs::accumulate(FWD(x), -21) + rs::accumulate(FWD(y), 0);
        };
        constexpr auto g = [](auto&& x, auto&& y) {
            return rs::accumulate(FWD(x), -60) + rs::accumulate(FWD(y), 0);
        };
        constexpr auto h = [](auto&& x, auto&& y) {
            return rs::accumulate(FWD(x), -22) + rs::accumulate(FWD(y), 0);
        };

        auto r = transform(std::tuple{f, g, h}, s, t);

        auto&& [x, y, z] = r;
        REQUIRE(x == 0);
        REQUIRE(y == 0);
        REQUIRE(z == 0);
    }
}

TEST_CASE("nested transform")
{
    using T = std::vector<int>;

    auto s = std::tuple{std::tuple{T{1, 2}}, std::tuple{T{3, 4}, T{5}}};

    auto a = transform(
        [](const T& v) { return vs::zip_with(std::plus{}, vs::all(v), vs::repeat(1)); },
        s);

    REQUIRE(rs::equal(get<0>(get<0>(a)), T{2, 3}));
    REQUIRE(rs::equal(get<0>(get<1>(a)), T{4, 5}));
    REQUIRE(rs::equal(get<1>(get<1>(a)), T{6}));

    auto t = std::tuple{std::tuple{T{0, 1}}, std::tuple{T{2, 3}, T{4}}};

    auto b = transform(
        [](const T& u, const T& v) {
            return vs::zip_with(std::plus{}, vs::all(u), vs::all(v));
        },
        s,
        t);

    REQUIRE(rs::equal(get<0>(get<0>(b)), T{1, 3}));
    REQUIRE(rs::equal(get<0>(get<1>(b)), T{5, 7}));
    REQUIRE(rs::equal(get<1>(get<1>(b)), T{9}));
}

TEST_CASE("transform_reduce")
{
    using T = std::vector<int>;
    auto s = std::tuple{T{0, 1, 2}, T{-1, 0, 0}, T{1, -10, 1}};

    auto k = transform_reduce(rs::max, s, std::plus{}, -3);

    REQUIRE(k == 0);

    auto [min, max] = transform_reduce(
        rs::minmax,
        s,
        [](auto&& acc, auto&& item) {
            return rs::minmax_result<int>{rs::min(acc.min, item.min),
                                          rs::max(acc.max, item.max)};
        },
        rs::minmax_result<int>{});

    REQUIRE(min == -10);
    REQUIRE(max == 2);
}

TEST_CASE("range lift")
{
    using T = std::vector<int>;

    const auto t = T{1, 2, 3};
    const auto s = T{4, 5, 6};

    constexpr auto plus = lift(std::plus{});

    auto r = plus(s, t);
    REQUIRE(rs::equal(r, T{5, 7, 9}));
}

TEST_CASE("tuple lift")
{
    using T = std::vector<int>;

    auto s = std::tuple{std::tuple{T{1, 2}}, std::tuple{T{3, 4}, T{5}}};
    constexpr auto lift1 = lift([](auto&& arg) { return arg + 1; });

    auto a = lift1(s);

    REQUIRE(rs::equal(get<0>(get<0>(a)), T{2, 3}));
    REQUIRE(rs::equal(get<0>(get<1>(a)), T{4, 5}));
    REQUIRE(rs::equal(get<1>(get<1>(a)), T{6}));

    auto t = std::tuple{std::tuple{T{0, 1}}, std::tuple{T{2, 3}, T{4}}};

    constexpr auto lift2 = lift([](auto&&... args) { return (args + ...); });
    auto b = lift2(s, t);

    REQUIRE(rs::equal(get<0>(get<0>(b)), T{1, 3}));
    REQUIRE(rs::equal(get<0>(get<1>(b)), T{5, 7}));
    REQUIRE(rs::equal(get<1>(get<1>(b)), T{9}));

    constexpr auto lift_plus = lift(std::plus{});
    auto c = lift_plus(s, t);

    REQUIRE(rs::equal(get<0>(get<0>(c)), T{1, 3}));
    REQUIRE(rs::equal(get<0>(get<1>(c)), T{5, 7}));
    REQUIRE(rs::equal(get<1>(get<1>(c)), T{9}));
}

TEST_CASE("resize_and_copy vector")
{
    using T = std::vector<int>;

    auto t = T{};
    resize_and_copy(t, vs::iota(0, 10));
    REQUIRE(rs::size(t) == rs::size(vs::iota(0, 10)));
    REQUIRE(rs::equal(t, vs::iota(0, 10)));

    resize_and_copy(t, vs::iota(0, 2));
    REQUIRE(rs::size(t) == rs::size(vs::iota(0, 2)));
    REQUIRE(rs::equal(t, vs::iota(0, 2)));

    resize_and_copy(vs::all(t), vs::iota(5, 10));
    REQUIRE(rs::size(t) == 5u);
    REQUIRE(rs::equal(t, vs::iota(5, 10)));

    resize_and_copy(t, 0);
    REQUIRE(rs::size(t) == 5u);
    REQUIRE(rs::equal(t, T{0, 0, 0, 0, 0}));
}

template <auto I>
struct loc_fn {
    constexpr auto operator()(auto&& loc)
    {
        auto&& [x, y, z] = loc;
        return x * y * z * std::get<I>(loc);
    }
};

TEST_CASE("resize_and_copy span")
{
    using T = std::span<int>;
    auto t_ = std::vector<int>(5);

    T t = t_;

    resize_and_copy(t, vs::iota(0, 10));
    REQUIRE(rs::size(t) == 5u);
    REQUIRE(rs::equal(t_, vs::iota(0, 5)));

    resize_and_copy(t, -1);
    REQUIRE(rs::size(t) == 5u);
    REQUIRE(rs::equal(t, vs::repeat_n(-1, rs::size(t))));

    resize_and_copy(t, vs::iota(0, 10) | vs::transform([](auto&& i) { return i + 1; }));
    REQUIRE(rs::size(t) == 5u);
    REQUIRE(rs::equal(t, vs::iota(1, 6)));
}

TEST_CASE("resize_and_copy tuples")
{
    using T = std::vector<int>;

    std::tuple<T, T> x{};
    {
        resize_and_copy(x, vs::iota(0, 10));
        auto&& [a, b] = x;
        REQUIRE(rs::size(a) == 10u);
        REQUIRE(rs::size(b) == 10u);
        REQUIRE(rs::equal(a, vs::iota(0, 10)));
        REQUIRE(rs::equal(a, b));

        resize_and_copy(x, -1);
        REQUIRE(rs::size(a) == 10u);
        REQUIRE(rs::size(b) == 10u);
        REQUIRE(rs::equal(a, vs::repeat_n(-1, rs::size(a))));
        REQUIRE(rs::equal(a, b));
    }

    {
        auto u = T{};
        auto v = T{};
        std::tuple<T&, T&> y{u, v};
        resize_and_copy(y, vs::iota(0, 10));
        auto&& [a, b] = y;
        REQUIRE(rs::size(a) == 10u);
        REQUIRE(rs::size(b) == 10u);
        REQUIRE(rs::equal(a, vs::iota(0, 10)));
        REQUIRE(rs::equal(a, b));

        resize_and_copy(y, -1);
        REQUIRE(rs::size(a) == 10u);
        REQUIRE(rs::size(b) == 10u);
        REQUIRE(rs::equal(a, vs::repeat_n(-1, rs::size(a))));
        REQUIRE(rs::equal(a, b));
    }

    {
        auto u = T(3);
        auto v = T(4);
        std::tuple<std::span<int>, std::span<int>> y{u, v};
        resize_and_copy(y, vs::iota(0, 10));
        auto&& [a, b] = y;
        REQUIRE(rs::size(a) == 3u);
        REQUIRE(rs::size(b) == 4u);
        REQUIRE(rs::equal(a, vs::iota(0, 3)));
        REQUIRE(rs::equal(b, vs::iota(0, 4)));

        resize_and_copy(y, -1);
        REQUIRE(rs::size(a) == 3u);
        REQUIRE(rs::size(b) == 4u);
        REQUIRE(rs::equal(a, vs::repeat_n(-1, rs::size(a))));
        REQUIRE(rs::equal(b, vs::repeat_n(-1, rs::size(b))));
    }
}

TEST_CASE("resize_and_copy tuples to tuples")
{
    using T = std::vector<int>;

    {
        std::tuple<T, T> x{};
        auto y = std::tuple{vs::iota(0, 10), vs::iota(3, 6)};

        resize_and_copy(x, y);
        auto&& [a, b] = x;
        REQUIRE(rs::size(a) == 10u);
        REQUIRE(rs::size(b) == 3u);
        REQUIRE(rs::equal(a, vs::iota(0, 10)));
        REQUIRE(rs::equal(b, vs::iota(3, 6)));
    }

    {
        auto u = T{};
        auto v = T{};
        std::tuple<T&, T&> x{u, v};
        auto y = std::tuple{vs::iota(0, 10), vs::iota(3, 6)};

        resize_and_copy(x, y);
        auto&& [a, b] = x;
        REQUIRE(rs::size(a) == 10u);
        REQUIRE(rs::size(b) == 3u);
        REQUIRE(rs::equal(a, vs::iota(0, 10)));
        REQUIRE(rs::equal(b, vs::iota(3, 6)));
    }

    {
        auto u = T{};
        auto v = T{};
        auto x = std::tuple{vs::all(u), vs::all(v)};

        resize_and_copy(x, std::tuple{vs::iota(0, 10), vs::iota(3, 6)});
        auto&& [a, b] = x;
        REQUIRE(rs::size(a) == 10u);
        REQUIRE(rs::size(b) == 3u);
        REQUIRE(rs::equal(a, vs::iota(0, 10)));
        REQUIRE(rs::equal(b, vs::iota(3, 6)));
    }
}

TEST_CASE("to_tuple")
{
    const auto i = vs::iota(0, 10);
    const auto j = vs::iota(1, 5);
    const auto k = vs::iota(10, 30);

    using T = std::vector<real>;
    {
        auto t = T{1, 2, 3};
        auto s = to<std::span<const real>>(t);
        REQUIRE(rs::equal(t, s));
    }

    {
        auto t = to<std::vector<int>>(i);
        REQUIRE(rs::equal(t, i));
    }

    {
        auto t = to<std::tuple<T>>(std::tuple{i});
        REQUIRE(rs::equal(get<0>(t), i));
    }

    {
        auto t = to<std::tuple<T, T>>(std::tuple{i, j});
        REQUIRE(rs::equal(get<0>(t), i));
        REQUIRE(rs::equal(get<1>(t), j));
    }

    {
        auto t = to<std::tuple<std::tuple<T>, std::tuple<T, T>>>(
            std::tuple{std::tuple{i}, std::tuple{j, k}});
        REQUIRE(rs::equal(get<0>(get<0>(t)), i));
        REQUIRE(rs::equal(get<0>(get<1>(t)), j));
        REQUIRE(rs::equal(get<1>(get<1>(t)), k));
    }

    {
        auto t = to<real3>(std::tuple{1.0, 2., 3.});
        REQUIRE(t == real3{1, 2, 3});
    }
}

TEST_CASE("make_tuple")
{
    auto s = make_tuple<std::tuple<int, int>>(5.1, 4.2);
    static_assert(std::same_as<std::tuple<double, double>, decltype(s)>);
    REQUIRE(get<0>(s) == 5.1);
    REQUIRE(get<1>(s) == 4.2);

    auto q = make_tuple<std::tuple<int>>(5.1, 4.2);
    static_assert(std::same_as<std::tuple<double, double>, decltype(q)>);
    REQUIRE(get<0>(q) == 5.1);
    REQUIRE(get<1>(q) == 4.2);
}

TEST_CASE("tuple_cat")
{
    auto s = std::tuple<int, int>{-1, -2};
    auto t = std::tuple<real>{0.1};
    auto u = std::tuple<std::tuple<unsigned>>{1u};

    auto v = tuple_cat<std::tuple<void*>>(s, t, u);

    static_assert(
        std::same_as<std::tuple<int, int, real, std::tuple<unsigned>>, decltype(v)>);

    auto&& [a, b, c, d] = v;
    REQUIRE(a == -1);
    REQUIRE(b == -2);
    REQUIRE(c == 0.1);
    REQUIRE(get<0>(d) == 1u);
}

TEST_CASE("join")
{
    auto s = std::tuple<int, int>{-1, -2};
    auto t = std::tuple<real>{0.1};
    auto u = std::tuple<std::tuple<unsigned>>{1u};

    // single nested tuple
    {
        auto v = std::tuple<std::tuple<int, int>>{s};
        auto j = join(v);
        static_assert(std::same_as<decltype(j), std::tuple<int, int>>);
        auto&& [a, b] = j;
        REQUIRE(a == -1);
        REQUIRE(b == -2);
    }

    {
        auto v = std::tuple{s, t, u};
        auto j = join(v);
        static_assert(
            std::same_as<decltype(j), std::tuple<int, int, real, std::tuple<unsigned>>>);

        auto&& [a, b, c, d] = j;
        REQUIRE(a == -1);
        REQUIRE(b == -2);
        REQUIRE(c == 0.1);
        REQUIRE(get<0>(d) == 1u);
    }
}
