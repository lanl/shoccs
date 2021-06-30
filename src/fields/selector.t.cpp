#include "selector.hpp"

#include <catch2/catch_test_macros.hpp>

#include <range/v3/all.hpp>

#include <vector>

using namespace ccs;

constexpr auto plus = lift(std::plus{});
constexpr auto dble = [](auto&& v) { return plus(v, v); };

TEST_CASE("planes construction")
{
    index_extents t{};
    auto x0 = sel::xmin(t);
    auto x1 = sel::xmax(t);
    auto y0 = sel::ymin(t);
    auto y1 = sel::ymax(t);
    auto z0 = sel::zmin(t);
    auto z1 = sel::zmax(t);

    static_assert(ViewClosure<decltype(x0)>);
    static_assert(ViewClosure<decltype(x1)>);
    static_assert(ViewClosure<decltype(y0)>);
    static_assert(ViewClosure<decltype(y1)>);
    static_assert(ViewClosure<decltype(z0)>);
    static_assert(ViewClosure<decltype(z1)>);
}

TEST_CASE("planes extraction")
{
    index_extents i{.extents = int3{2, 3, 4}};
    auto t = tuple{vs::iota(0, 2 * 3 * 4)};

    REQUIRE(rs::equal(t | sel::xmin(i), vs::iota(0, 12)));
    REQUIRE(rs::equal(t | sel::xmax(i), vs::iota(12, 24)));
    // == doesn't work with ymin/max selector...
    REQUIRE(rs::equal(t | sel::ymin(i), tuple{std::vector{0, 1, 2, 3, 12, 13, 14, 15}}));
    REQUIRE(
        rs::equal(t | sel::ymax(i), tuple{std::vector{8, 9, 10, 11, 20, 21, 22, 23}}));
    REQUIRE(rs::equal(t | sel::zmin(i), std::vector{0, 4, 8, 12, 16, 20}));
    REQUIRE(rs::equal(t | sel::zmax(i), tuple{std::vector{3, 7, 11, 15, 19, 23}}));

    auto u = tuple<std::vector<int>>{vs::iota(24, 48)};
    REQUIRE(rs::equal(get<0>(t | sel::xmin(i)).apply(u), u | sel::xmin(i)));
    REQUIRE(rs::equal(get<0>(t | sel::ymax(i)).apply(u), u | sel::ymax(i)));
    REQUIRE(rs::equal(get<0>(t | sel::zmin(i)).apply(u), u | sel::zmin(i)));
}

TEST_CASE("planes assignment")
{
    using T = std::vector<int>;
    index_extents i{.extents = int3{2, 3, 4}};
    auto t = tuple<T>{vs::iota(0, 2 * 3 * 4)};

    t | sel::xmin(i) = -1;

    REQUIRE((t | sel::xmin(i)) == tuple{vs::repeat_n(-1, 12)});
    REQUIRE((t | sel::xmax(i)) == tuple{vs::iota(12, 24)});

    t | sel::xmax(i) = -2;

    REQUIRE((t | sel::xmin(i)) == tuple{vs::repeat_n(-1, 12)});
    REQUIRE((t | sel::xmax(i)) == tuple{vs::repeat_n(-2, 12)});

    t | sel::zmin(i) = vs::iota(0) | vs::stride(4) | vs::take_exactly(6);
    REQUIRE((t | sel::zmin(i)) == tuple{std::vector{0, 4, 8, 12, 16, 20}});

    t | sel::ymax(i) = -3;
    REQUIRE((t | sel::zmin(i)) == tuple{std::vector{0, 4, -3, 12, 16, -3}});
    REQUIRE(rs::equal(t | sel::ymax(i), vs::repeat_n(-3, 8)));
}

TEST_CASE("planes scalar extraction")
{
    using T = std::vector<int>;
    index_extents i{.extents = int3{2, 3, 4}};
    scalar<T> s{tuple{vs::iota(0, 24)}, tuple{0, 0, 0}};

    REQUIRE((s | sel::xmin(i)) == tuple{vs::iota(0, 12)});
    REQUIRE((s | sel::xmax(i)) == tuple{vs::iota(12, 24)});
    // == doesn't work with ymin/max selector...
    REQUIRE(rs::equal(s | sel::ymin(i), tuple{T{0, 1, 2, 3, 12, 13, 14, 15}}));
    REQUIRE(rs::equal(s | sel::ymax(i), tuple{T{8, 9, 10, 11, 20, 21, 22, 23}}));
    REQUIRE((s | sel::zmin(i)) == tuple{T{0, 4, 8, 12, 16, 20}});
    REQUIRE((s | sel::zmax(i)) == tuple{T{3, 7, 11, 15, 19, 23}});

    scalar<T> v{tuple{vs::iota(24, 48)}, tuple{0, 0, 0}};

    REQUIRE(rs::equal(get<0>(s | sel::xmin(i)).apply(v), v | sel::xmin(i)));
    REQUIRE(rs::equal(get<0>(s | sel::ymax(i)).apply(v), v | sel::ymax(i)));
    REQUIRE(rs::equal(get<0>(s | sel::zmin(i)).apply(v), v | sel::zmin(i)));
}

TEST_CASE("planes scalar assignment")
{
    using T = std::vector<int>;
    index_extents i{.extents = int3{2, 3, 4}};
    scalar<T> s{tuple{vs::iota(0, 24)}, tuple{0, 0, 0}};

    s | sel::xmin(i) = -1;

    REQUIRE((s | sel::xmin(i)) == tuple{vs::repeat_n(-1, 12)});
    REQUIRE((s | sel::xmax(i)) == tuple{vs::iota(12, 24)});

    s | sel::xmax(i) = -2;

    REQUIRE((s | sel::xmin(i)) == tuple{vs::repeat_n(-1, 12)});
    REQUIRE((s | sel::xmax(i)) == tuple{vs::repeat_n(-2, 12)});

    s | sel::zmin(i) = vs::iota(0) | vs::stride(4) | vs::take_exactly(6);
    REQUIRE((s | sel::zmin(i)) == tuple{T{0, 4, 8, 12, 16, 20}});

    s | sel::ymax(i) = -3;
    REQUIRE((s | sel::zmin(i)) == tuple{T{0, 4, -3, 12, 16, -3}});
    REQUIRE(rs::equal(s | sel::ymax(i), vs::repeat_n(-3, 8)));

    scalar<T> v{tuple{vs::iota(24, 48)}, tuple{0, 0, 0}};
    s | sel::xmin(i) = v;

    REQUIRE((s | sel::xmin(i)) == tuple{vs::iota(24, 36)});
    REQUIRE((s | sel::zmin(i)) == tuple{T{24, 28, 32, 12, 16, -3}});
}

TEST_CASE("planes vector extraction")
{
    using T = std::vector<int>;
    index_extents i{.extents = int3{2, 3, 4}};
    vector<T> v{tuple{tuple{vs::iota(0, 24)}, tuple{0, 0, 0}},
                tuple{tuple{vs::iota(24, 48)}, tuple{0, 0, 0}},
                tuple{tuple{vs::iota(48, 72)}, tuple{0, 0, 0}}};

    REQUIRE(get<0>(v | sel::xmin(i)) == tuple{vs::iota(0, 12)});
    REQUIRE(get<1>(v | sel::xmax(i)) == tuple{vs::iota(36, 48)});
    REQUIRE(rs::equal(get<2>(v | sel::ymin(i)), T{48, 49, 50, 51, 60, 61, 62, 63}));
    REQUIRE(rs::equal(get<0>(v | sel::ymax(i)), T{8, 9, 10, 11, 20, 21, 22, 23}));
    REQUIRE((get<1>(v | sel::zmin(i)) == tuple{T{24, 28, 32, 36, 40, 44}}));
    REQUIRE((get<2>(v | sel::zmax(i)) == tuple{T{51, 55, 59, 63, 67, 71}}));

    vector<T> u{tuple{tuple{vs::iota(48, 72)}, tuple{0, 0, 0}},
                tuple{tuple{vs::iota(0, 24)}, tuple{0, 0, 0}},
                tuple{tuple{vs::iota(24, 48)}, tuple{0, 0, 0}}};

    REQUIRE(rs::equal(get<0>(v | sel::xmin(i)).apply(u), get<0>(u | sel::xmin(i))));
    REQUIRE(rs::equal(get<1>(v | sel::ymax(i)).apply(u), get<1>(u | sel::ymax(i))));
    REQUIRE(rs::equal(get<2>(v | sel::zmin(i)).apply(u), get<2>(u | sel::zmin(i))));
}

TEST_CASE("planes vector assignment")
{
    using T = std::vector<int>;
    index_extents i{.extents = int3{2, 3, 4}};
    vector<T> v{tuple{tuple{vs::iota(0, 24)}, tuple{0, 0, 0}},
                tuple{tuple{vs::iota(24, 48)}, tuple{0, 0, 0}},
                tuple{tuple{vs::iota(48, 72)}, tuple{0, 0, 0}}};

    v | sel::xmin(i) = -1;

    REQUIRE(get<0>(v | sel::xmin(i)) == tuple{vs::repeat_n(-1, 12)});
    REQUIRE(get<1>(v | sel::xmax(i)) == tuple{vs::iota(36, 48)});
    REQUIRE(get<2>(v | sel::xmin(i)) == tuple{vs::repeat_n(-1, 12)});

    vector<T> u{tuple{tuple{vs::iota(48, 72)}, tuple{0, 0, 0}},
                tuple{tuple{vs::iota(0, 24)}, tuple{0, 0, 0}},
                tuple{tuple{vs::iota(24, 48)}, tuple{0, 0, 0}}};

    auto w = v | sel::ymin(i);
    static_assert(!SimilarTuples<decltype(w), vector<T>>);

    v | sel::ymin(i) = u;

    REQUIRE(rs::equal(get<0>(v | sel::ymin(i)), get<0>(u | sel::ymin(i))));
    REQUIRE(rs::equal(get<1>(v | sel::ymin(i)), get<1>(u | sel::ymin(i))));
    REQUIRE(rs::equal(get<2>(v | sel::ymin(i)), get<2>(u | sel::ymin(i))));

    REQUIRE(get<1>(v | sel::zmin(i)) == tuple{T{0, -1, -1, 12, 40, 44}});
}

TEST_CASE("multi_slice construction")
{
    std::vector<index_slice> slices{};
    auto s = sel::multi_slice(slices);

    static_assert(ViewClosure<decltype(s)>);
}

TEST_CASE("multi_slice extraction")
{
    auto t = tuple{vs::iota(0, 24)};

    SECTION("whole container")
    {
        std::vector<index_slice> slices{{0, 24}};
        auto s = t | sel::multi_slice(slices);
        REQUIRE(rs::size(s) == 24);
        REQUIRE(rs::equal(t | sel::multi_slice(slices), t));
    }

    SECTION("whole container multiple slices")
    {
        std::vector<index_slice> slices{{0, 5}, {5, 10}, {10, 11}, {11, 24}};
        auto s = t | sel::multi_slice(slices);
        REQUIRE(rs::size(s) == 24);
        REQUIRE(rs::equal(t | sel::multi_slice(slices), t));
    }

    SECTION("partial container, single slice")
    {
        std::vector<index_slice> slices{{2, 10}};
        auto s = t | sel::multi_slice(slices);
        REQUIRE(rs::size(s) == 8);
        REQUIRE(rs::equal(s, vs::iota(2, 10)));
    }

    SECTION("partial container, many slices")
    {
        std::vector<index_slice> slices{{1, 4}, {5, 6}, {10, 23}};
        auto s = t | sel::multi_slice(slices);
        REQUIRE(rs::size(s) == 17);
        REQUIRE(
            rs::equal(s, vs::concat(vs::iota(1, 4), vs::iota(5, 6), vs::iota(10, 23))));

        auto q = tuple{vs::iota(24, 48)};
        auto z = get<0>(s).apply(q);
        REQUIRE(rs::size(z) == 17);
        REQUIRE(rs::equal(z, q | sel::multi_slice(slices)));
    }
}

TEST_CASE("multi_slice assignment")
{
    using T = std::vector<int>;
    auto t = tuple<T>{vs::iota(0, 24)};

    std::vector<index_slice> a{{0, 24}};
    auto whole = sel::multi_slice(a);

    t | whole = -1;

    REQUIRE(t == tuple{vs::repeat_n(-1, 24)});

    std::vector<index_slice> b{{1, 8}, {22, 24}};
    auto sparse0 = sel::multi_slice(b);

    t | sparse0 = -2;
    REQUIRE(t == tuple{vs::concat(vs::repeat_n(-1, 1),
                                  vs::repeat_n(-2, 7),
                                  vs::repeat_n(-1, 14),
                                  vs::repeat_n(-2, 2))});

    t | sparse0 = vs::concat(vs::iota(1, 8), vs::iota(22, 24));
    REQUIRE(t == tuple{vs::concat(vs::repeat_n(-1, 1),
                                  vs::iota(1, 8),
                                  vs::repeat_n(-1, 14),
                                  vs::iota(22, 24))});
    tuple<T> t2 = t + 1;

    t | sparse0 = t2 | sparse0;
    REQUIRE(t == tuple{vs::concat(vs::repeat_n(-1, 1),
                                  vs::iota(2, 9),
                                  vs::repeat_n(-1, 14),
                                  vs::iota(23, 25))});
}

TEST_CASE("mulit_slice scalar extraction")
{
    using T = std::vector<int>;
    using U = std::vector<index_slice>;

    scalar<T> s{tuple{vs::iota(0, 24)}, tuple{0, 0, 0}};

    U whole{{0, 24}};
    REQUIRE(rs::equal(s | sel::multi_slice(whole), s | sel::D));

    U sparse{{1, 2}, {4, 8}, {20, 24}};
    REQUIRE(rs::equal(s | sel::multi_slice(sparse),
                      vs::concat(vs::iota(1, 2), vs::iota(4, 8), vs::iota(20, 24))));

    scalar<T> t{tuple{vs::iota(24, 48)}, tuple{0, 0, 0}};
    REQUIRE(rs::equal(get<0>(s | sel::multi_slice(sparse)).apply(t),
                      t | sel::multi_slice(sparse)));
}

TEST_CASE("multi_slice scalar assignment")
{
    using T = std::vector<int>;
    using U = std::vector<index_slice>;

    scalar<T> s{tuple{vs::iota(0, 24)}, tuple{0, 0, 0}};

    U a{{0, 24}};
    auto whole = sel::multi_slice(a);

    s | whole = -1;

    REQUIRE(get<0>(s) == tuple{vs::repeat_n(-1, 24)});

    U b{{1, 3}, {6, 12}, {22, 24}};
    auto sparse = sel::multi_slice(b);

    s | sparse = -2;
    REQUIRE(get<0>(s) == tuple{vs::concat(vs::repeat_n(-1, 1),
                                          vs::repeat_n(-2, 2),
                                          vs::repeat_n(-1, 3),
                                          vs::repeat_n(-2, 6),
                                          vs::repeat_n(-1, 10),
                                          vs::repeat_n(-2, 2))});

    s | sparse = vs::concat(vs::iota(1, 3), vs::iota(6, 12), vs::iota(22, 24));
    REQUIRE(get<0>(s) == tuple{vs::concat(vs::repeat_n(-1, 1),
                                          vs::iota(1, 3),
                                          vs::repeat_n(-1, 3),
                                          vs::iota(6, 12),
                                          vs::repeat_n(-1, 10),
                                          vs::iota(22, 24))});
    scalar<T> s2 = s + 1;

    s | sparse = s2; // or s | spare = s2 | sparse;
    REQUIRE(get<0>(s) == tuple{vs::concat(vs::repeat_n(-1, 1),
                                          vs::iota(2, 4),
                                          vs::repeat_n(-1, 3),
                                          vs::iota(7, 13),
                                          vs::repeat_n(-1, 10),
                                          vs::iota(23, 25))});
}

TEST_CASE("multi_slice vector extraction")
{
    using T = std::vector<int>;
    using U = std::vector<index_slice>;

    vector<T> v{tuple{tuple{vs::iota(0, 24)}, tuple{0, 0, 0}},
                tuple{tuple{vs::iota(24, 48)}, tuple{0, 0, 0}},
                tuple{tuple{vs::iota(48, 72)}, tuple{0, 0, 0}}};

    U whole{{0, 24}};
    REQUIRE(rs::equal(get<0>(v | sel::multi_slice(whole)), v | sel::Dx));
    REQUIRE(rs::equal(get<1>(v | sel::multi_slice(whole)), v | sel::Dy));
    REQUIRE(rs::equal(get<2>(v | sel::multi_slice(whole)), v | sel::Dz));

    U sparse{{1, 2}, {4, 8}, {20, 24}};
    REQUIRE(rs::equal(get<0>(v | sel::multi_slice(sparse)),
                      get<0>(v) | sel::multi_slice(sparse)));
    REQUIRE(rs::equal(get<1>(v | sel::multi_slice(sparse)),
                      get<1>(v) | sel::multi_slice(sparse)));
    REQUIRE(rs::equal(get<2>(v | sel::multi_slice(sparse)),
                      get<2>(v) | sel::multi_slice(sparse)));

    vector<T> u{tuple{tuple{vs::iota(48, 72)}, tuple{0, 0, 0}},
                tuple{tuple{vs::iota(0, 24)}, tuple{0, 0, 0}},
                tuple{tuple{vs::iota(24, 48)}, tuple{0, 0, 0}}};

    const auto sp = sel::multi_slice(sparse);

    REQUIRE(rs::equal(get<0>(v | sp).apply(u), get<0>(u) | sp));
    REQUIRE(rs::equal(get<1>(v | sp).apply(u), get<1>(u) | sp));
    REQUIRE(rs::equal(get<2>(v | sp).apply(u), get<2>(u) | sp));
}

TEST_CASE("multi_slice vector assignment")
{
    using T = std::vector<int>;
    using U = std::vector<index_slice>;

    vector<T> v{tuple{tuple{vs::iota(0, 24)}, tuple{0, 0, 0}},
                tuple{tuple{vs::iota(24, 48)}, tuple{0, 0, 0}},
                tuple{tuple{vs::iota(48, 72)}, tuple{0, 0, 0}}};

    U a{{0, 24}};
    auto whole = sel::multi_slice(a);

    v | whole = -1;
    REQUIRE(get<vi::Dx>(v) == tuple{vs::repeat_n(-1, 24)});
    REQUIRE(get<vi::Dy>(v) == tuple{vs::repeat_n(-1, 24)});
    REQUIRE(get<vi::Dz>(v) == tuple{vs::repeat_n(-1, 24)});

    U b{{1, 3}, {6, 12}, {22, 24}};
    auto sparse = sel::multi_slice(b);

    v | sparse = -2;
    REQUIRE(get<vi::Dx>(v) == tuple{vs::concat(vs::repeat_n(-1, 1),
                                               vs::repeat_n(-2, 2),
                                               vs::repeat_n(-1, 3),
                                               vs::repeat_n(-2, 6),
                                               vs::repeat_n(-1, 10),
                                               vs::repeat_n(-2, 2))});
    REQUIRE(get<vi::Dx>(v) == get<vi::Dy>(v));
    REQUIRE(get<vi::Dx>(v) == get<vi::Dz>(v));

    v | sparse = vs::concat(vs::iota(1, 3), vs::iota(6, 12), vs::iota(22, 24));
    REQUIRE(get<vi::Dx>(v) == tuple{vs::concat(vs::repeat_n(-1, 1),
                                               vs::iota(1, 3),
                                               vs::repeat_n(-1, 3),
                                               vs::iota(6, 12),
                                               vs::repeat_n(-1, 10),
                                               vs::iota(22, 24))});
    REQUIRE(get<vi::Dx>(v) == get<vi::Dy>(v));
    REQUIRE(get<vi::Dx>(v) == get<vi::Dz>(v));

    v | sparse = tuple{vs::concat(vs::iota(1, 3), vs::iota(6, 12), vs::iota(22, 24)),
                       vs::concat(vs::iota(25, 27), vs::iota(30, 36), vs::iota(46, 48)),
                       vs::concat(vs::iota(49, 51), vs::iota(54, 60), vs::iota(70, 72))};
    REQUIRE(get<vi::Dx>(v) == tuple{vs::concat(vs::repeat_n(-1, 1),
                                               vs::iota(1, 3),
                                               vs::repeat_n(-1, 3),
                                               vs::iota(6, 12),
                                               vs::repeat_n(-1, 10),
                                               vs::iota(22, 24))});
    REQUIRE(get<vi::Dy>(v) == tuple{vs::concat(vs::repeat_n(-1, 1),
                                               vs::iota(25, 27),
                                               vs::repeat_n(-1, 3),
                                               vs::iota(30, 36),
                                               vs::repeat_n(-1, 10),
                                               vs::iota(46, 48))});
    REQUIRE(get<vi::Dz>(v) == tuple{vs::concat(vs::repeat_n(-1, 1),
                                               vs::iota(49, 51),
                                               vs::repeat_n(-1, 3),
                                               vs::iota(54, 60),
                                               vs::repeat_n(-1, 10),
                                               vs::iota(70, 72))});

    vector<T> u = v + 1;

    v | sparse = u;

    REQUIRE(get<vi::Dx>(v) == tuple{vs::concat(vs::repeat_n(-1, 1),
                                               vs::iota(2, 4),
                                               vs::repeat_n(-1, 3),
                                               vs::iota(7, 13),
                                               vs::repeat_n(-1, 10),
                                               vs::iota(23, 25))});
    REQUIRE(get<vi::Dy>(v) == tuple{vs::concat(vs::repeat_n(-1, 1),
                                               vs::iota(26, 28),
                                               vs::repeat_n(-1, 3),
                                               vs::iota(31, 37),
                                               vs::repeat_n(-1, 10),
                                               vs::iota(47, 49))});
    REQUIRE(get<vi::Dz>(v) == tuple{vs::concat(vs::repeat_n(-1, 1),
                                               vs::iota(50, 52),
                                               vs::repeat_n(-1, 3),
                                               vs::iota(55, 61),
                                               vs::repeat_n(-1, 10),
                                               vs::iota(71, 73))});
}

TEST_CASE("default operators for storing selections in mesh")
{

    using F_t = sel::multi_slice_t;
    // default constructible;
    F_t F{};
    auto v = std::vector<index_slice>{{1, 4}};
    // move assignable

    F = sel::multi_slice(v);
}

TEST_CASE("optional tuple")
{
    using T = std::vector<int>;
    T t{1, 2, 3};

    REQUIRE(rs::size(t | sel::optional_view(false)) == 0);
    REQUIRE(rs::size(t | sel::optional_view(true)) == 3);

    index_extents i{.extents = int3{2, 3, 4}};
    resize_and_copy(t, vs::iota(0, 2 * 3 * 4));

    auto x = tuple{t};
    REQUIRE(rs::size(x | sel::optional_view(false)) == 0);
    REQUIRE(rs::size(x | sel::optional_view(true)) == rs::size(x));

    REQUIRE(rs::size(x | sel::xmin(i) | sel::optional_view(false)) == 0);
    REQUIRE(rs::size(x | sel::xmin(i) | sel::optional_view(true)) == 12u);

    auto y = x | sel::xmin(i) | sel::optional_view(false);
    y = 0;
    REQUIRE(rs::equal(x, vs::iota(0, 24)));

    auto z = x | sel::zmax(i) | sel::optional_view(true);
    z = 0;
    REQUIRE((x | sel::zmin(i)) == tuple{T{0, 4, 8, 12, 16, 20}});
    REQUIRE((x | sel::zmax(i)) == tuple{T{0, 0, 0, 0, 0, 0}});
}

TEST_CASE("optional scalar")
{
    using T = std::vector<int>;
    index_extents i{.extents = int3{2, 3, 4}};
    scalar<T> s{tuple{vs::iota(0, 24)}, tuple{0, 0, 0}};

    auto q = s | sel::xmin(i) | sel::optional_view(false);
    REQUIRE(rs::size(q) == 0);
    q = 0;
    REQUIRE((s | sel::xmin(i)) == tuple{vs::iota(0, 12)});
    q = vs::iota(-12, 0);
    REQUIRE((s | sel::xmin(i)) == tuple{vs::iota(0, 12)});

    auto r = s | tuple{sel::xmin(i) | sel::optional_view(false),
                       sel::xmax(i) | sel::optional_view(true),
                       sel::ymin(i) | sel::optional_view(false),
                       sel::ymax(i) | sel::optional_view(false),
                       sel::zmin(i) | sel::optional_view(false),
                       sel::zmax(i) | sel::optional_view(false)};
    // get<0>(r) = 0;
    r = 0;
    REQUIRE((s | sel::xmin(i)) == tuple{vs::iota(0, 12)});
    REQUIRE((s | sel::xmax(i)) == tuple{vs::repeat_n(0, 12)});

    s |
        tuple{sel::xmin(i),
              sel::xmax(i),
              sel::ymin(i),
              sel::ymax(i),
              sel::zmin(i),
              sel::zmax(i)} |
        tuple{sel::optional_view(false),
              sel::optional_view(true),
              sel::optional_view(false),
              sel::optional_view(false),
              sel::optional_view(false),
              sel::optional_view(false)} = 1;

    REQUIRE((s | sel::xmin(i)) == tuple{vs::iota(0, 12)});
    REQUIRE((s | sel::xmax(i)) == tuple{vs::repeat_n(1, 12)});

    scalar<T> v{tuple{vs::iota(24, 48)}, tuple{0, 0, 0}};

    s |
        tuple{sel::xmin(i),
              sel::xmax(i),
              sel::ymin(i),
              sel::ymax(i),
              sel::zmin(i),
              sel::zmax(i)} |
        tuple{sel::optional_view(false),
              sel::optional_view(true),
              sel::optional_view(false),
              sel::optional_view(false),
              sel::optional_view(false),
              sel::optional_view(false)} = v;
    REQUIRE((s | sel::xmin(i)) == tuple{vs::iota(0, 12)});
    REQUIRE((s | sel::xmax(i)) == tuple{vs::iota(36, 48)});
}

TEST_CASE("optional vector")
{
    using T = std::vector<int>;
    index_extents i{.extents = int3{2, 3, 4}};
    vector<T> v{tuple{tuple{vs::iota(0, 24)}, tuple{0, 0, 0}},
                tuple{tuple{vs::iota(24, 48)}, tuple{0, 0, 0}},
                tuple{tuple{vs::iota(48, 72)}, tuple{0, 0, 0}}};

    auto q = v | sel::xmin(i) | sel::optional_view(false);
    REQUIRE(rs::size(get<0>(q)) == 0);
    q = 0;
    REQUIRE(get<0>(v | sel::xmin(i)) == tuple{vs::iota(0, 12)});
    q = vs::iota(-12, 0);
    REQUIRE(get<1>(v | sel::xmin(i)) == tuple{vs::iota(24, 36)});

    auto r = v | tuple{sel::xmin(i) | sel::optional_view(false),
                       sel::xmax(i) | sel::optional_view(true),
                       sel::ymin(i) | sel::optional_view(false),
                       sel::ymax(i) | sel::optional_view(false),
                       sel::zmin(i) | sel::optional_view(false),
                       sel::zmax(i) | sel::optional_view(false)};

    r = 0;
    REQUIRE(get<2>(v | sel::xmin(i)) == tuple{vs::iota(48, 60)});
    REQUIRE(get<2>(v | sel::xmax(i)) == tuple{vs::repeat_n(0, 12)});

    using R = std::remove_cvref_t<decltype(r)>;
    REQUIRE(!NestedTuple<R>);
    REQUIRE(std::tuple_size_v<R> == 18);

    v | (tuple{sel::xmin(i),
               sel::xmax(i),
               sel::ymin(i),
               sel::ymax(i),
               sel::zmin(i),
               sel::zmax(i)} |
         tuple{sel::optional_view(false),
               sel::optional_view(true),
               sel::optional_view(false),
               sel::optional_view(false),
               sel::optional_view(false),
               sel::optional_view(false)}) = 1;

    REQUIRE(get<1>(v | sel::xmin(i)) == tuple{vs::iota(24, 36)});
    REQUIRE(get<1>(v | sel::xmax(i)) == tuple{vs::repeat_n(1, 12)});

    vector<T> u{tuple{tuple{vs::iota(48, 72)}, tuple{0, 0, 0}},
                tuple{tuple{vs::iota(0, 24)}, tuple{0, 0, 0}},
                tuple{tuple{vs::iota(24, 48)}, tuple{0, 0, 0}}};

    v | (tuple{sel::xmin(i),
               sel::xmax(i),
               sel::ymin(i),
               sel::ymax(i),
               sel::zmin(i),
               sel::zmax(i)} |
         tuple{sel::optional_view(false),
               sel::optional_view(true),
               sel::optional_view(false),
               sel::optional_view(false),
               sel::optional_view(false),
               sel::optional_view(false)}) = u;
    REQUIRE(get<0>(v | sel::xmin(i)) == tuple{vs::iota(0, 12)});
    REQUIRE(get<1>(v | sel::xmin(i)) == tuple{vs::iota(24, 36)});
    REQUIRE(get<2>(v | sel::xmin(i)) == tuple{vs::iota(48, 60)});
    REQUIRE(get<0>(v | sel::xmax(i)) == tuple{vs::iota(60, 72)});
    REQUIRE(get<1>(v | sel::xmax(i)) == tuple{vs::iota(12, 24)});
    REQUIRE(get<2>(v | sel::xmax(i)) == tuple{vs::iota(36, 48)});
}

TEST_CASE("multi_slice math")
{
    using T = std::vector<int>;
    using U = std::vector<index_slice>;

    scalar<T> s{tuple{vs::iota(0, 24)}, tuple{0, 0, 0}};

    U a{{0, 24}};
    auto whole = sel::multi_slice(a);

    s | whole += 1;

    REQUIRE(get<0>(s) == tuple{vs::iota(1, 25)});

    U b{{1, 3}, {6, 12}, {22, 24}};
    auto sparse = sel::multi_slice(b);
    s | sparse -= 1; // vs::iota(0, 24)

    REQUIRE(get<0>(s) == tuple{vs::concat(vs::iota(1, 2),   /* whole */
                                          vs::iota(1, 3),   /* sparse */
                                          vs::iota(4, 7),   /* whole */
                                          vs::iota(6, 12),  /* sparse */
                                          vs::iota(13, 23), /* whole */
                                          vs::iota(22, 24)  /* sparse */
                                          )});

    s | sparse += tuple{tuple{vs::iota(0, 24)},
                        tuple{vs::iota(0, 0), vs::iota(0, 0), vs::iota(0, 0)}};

    REQUIRE(get<0>(s) == tuple{vs::concat(vs::iota(1, 2),        /* whole */
                                          dble(vs::iota(1, 3)),  /* sparse */
                                          vs::iota(4, 7),        /* whole */
                                          dble(vs::iota(6, 12)), /* sparse */
                                          vs::iota(13, 23),      /* whole */
                                          dble(vs::iota(22, 24)) /* sparse */
                                          )});
}
