#include "TupleUtils.hpp"

#include <catch2/catch_test_macros.hpp>

#include <range/v3/algorithm/equal.hpp>
#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/range/concepts.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/repeat_n.hpp>
#include <range/v3/view/zip.hpp>

#include <vector>

TEST_CASE("for_each")
{
    using namespace ccs;
    using namespace field::tuple;

    using T = std::vector<int>;

    auto s = std::tuple{T{1, 2, 3}, T{4, 5, 6}, T{7, 8, 9}};
    auto t = std::tuple{T{4, 5, 6}, T{1, 2, 3}, T{-1, -2, -3}};

    for_each(
        [](auto&& v) {
            for (auto&& i : v) i += 1;
        },
        s);
    REQUIRE(rs::equal(get<0>(s), T{2, 3, 4}));
    REQUIRE(rs::equal(get<1>(s), T{5, 6, 7}));
    REQUIRE(rs::equal(get<2>(s), T{8, 9, 10}));

    for_each(
        [](auto&& x, auto&& y) {
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
        []<auto I>(traits::mp_size_t<I>, auto&& v) {
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
                        [](auto&& x, auto&& y) {
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

TEST_CASE("transform")
{
    using namespace ccs;
    using namespace field::tuple;

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
        constexpr auto f = []<auto I>(traits::mp_size_t<I>, auto&& vec)
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
        constexpr auto f = []<auto I>(traits::mp_size_t<I>, auto&&... vec)
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

TEST_CASE("resize_and_copy vector")
{
    using namespace ccs;
    using namespace field::tuple;

    using T = std::vector<int>;

    auto t = T{};
    resize_and_copy(t, vs::iota(0, 10));
    REQUIRE(rs::size(t) == rs::size(vs::iota(0, 10)));
    REQUIRE(rs::equal(t, vs::iota(0, 10)));

    resize_and_copy(t, vs::iota(0, 2));
    REQUIRE(rs::size(t) == rs::size(vs::iota(0, 2)));
    REQUIRE(rs::equal(t, vs::iota(0, 2)));

    resize_and_copy(vs::all(t), vs::iota(5, 1024));
    REQUIRE(rs::size(t) == 2u);
    REQUIRE(rs::equal(t, T{5, 6}));

    resize_and_copy(t, 0);
    REQUIRE(rs::size(t) == 2u);
    REQUIRE(rs::equal(t, T{0, 0}));
}

TEST_CASE("resize_and_copy span")
{
    using namespace ccs;
    using namespace field::tuple;

    using T = std::span<int>;
    auto t_ = std::vector<int>(5);

    T t = t_;

    resize_and_copy(t, vs::iota(0, 10));
    REQUIRE(rs::size(t) == 5u);
    REQUIRE(rs::equal(t_, vs::iota(0, 5)));

    resize_and_copy(t, -1);
    REQUIRE(rs::size(t) == 5u);
    REQUIRE(rs::equal(t, vs::repeat_n(-1, rs::size(t))));
}

TEST_CASE("resize_and_copy tuples")
{
    using namespace ccs;
    using namespace field::tuple;

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
    using namespace ccs;
    using namespace field::tuple;

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
        auto u = T(3);
        auto v = T(4);
        auto x = std::tuple{vs::all(u), vs::all(v)};

        resize_and_copy(x, std::tuple{vs::iota(0, 10), vs::iota(3, 6)});
        auto&& [a, b] = x;
        REQUIRE(rs::size(a) == 3u);
        REQUIRE(rs::size(b) == 4u);
        REQUIRE(rs::equal(a, T{0, 1, 2}));
        REQUIRE(rs::equal(b, T{3, 4, 5, 0}));
    }
}

TEST_CASE("to_tuple")
{
    using namespace ccs;
    using namespace field::tuple;

    std::tuple<int, int, int> x{to_tuple<std::tuple<int, int, int>>(std::tuple{0, 1, 2})};
    REQUIRE(get<0>(x) == 0);
    REQUIRE(get<1>(x) == 1);
    REQUIRE(get<2>(x) == 2);

    using T = std::vector<real>;
    auto y = to_tuple<std::tuple<T, T>>(std::tuple{T{1, 2, 3}, T{4, 5, 6}});
    REQUIRE(rs::equal(get<0>(y), T{1, 2, 3}));
    REQUIRE(rs::equal(get<1>(y), T{4, 5, 6}));

    const auto i = vs::iota(5, 20);
    const auto j = vs::iota(-100, 0);
    auto z = to_tuple<std::tuple<T, T>>(std::tuple{i, j});
    REQUIRE(rs::equal(get<0>(z), i));
    REQUIRE(rs::equal(get<1>(z), j));
}

TEST_CASE("makeTuple")
{
    using namespace ccs::field::tuple;

    auto s = makeTuple<std::tuple<int, int>>(5.1, 4.2);
    static_assert(std::same_as<std::tuple<double, double>, decltype(s)>);
    REQUIRE(get<0>(s) == 5.1);
    REQUIRE(get<1>(s) == 4.2);
}
