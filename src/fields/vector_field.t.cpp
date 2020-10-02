#include "select.hpp"
#include "vector_field.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <range/v3/algorithm/equal.hpp>
#include <range/v3/algorithm/fill.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/repeat_n.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/take_exactly.hpp>

TEST_CASE("range transforms")
{
    using namespace ccs;

    const auto vr = vector_range{vs::ints(0, 5), vs::ints(1, 7), vs::ints(-3, 0)};

    REQUIRE(vr.size() == int3{5, 6, 3});

    auto t = vs::transform([](auto&& i) { return i * i; });
    auto a = vr >> t;

    REQUIRE(Vector_Field<decltype(a)>);

    REQUIRE(rs::equal(a.x, vr.x | t));
    REQUIRE(rs::equal(a.y, vr.y | t));
    REQUIRE(rs::equal(a.z, vr.z | t));

    auto trim = vector_take(int3{2, 3, 1});
    auto v = vr >> trim;

    REQUIRE(v.size() == int3{2, 3, 1});
    REQUIRE(rs::equal(v.x, vs::ints(0, 2)));
    REQUIRE(rs::equal(v.y, vs::ints(1, 4)));
    REQUIRE(rs::equal(v.z, vs::ints(-3, -2)));

    {
        auto r = v >> [](auto&& a) { return a[0];};
        REQUIRE(r.x == 0);
        REQUIRE(r.y == 1);
        REQUIRE(r.z == -3);
    }
}

TEST_CASE("resizing")
{
    using namespace ccs;

    using T = std::vector<real>;

    auto v = vector_range{T{0, 1, 2}, T{-1, -2}, T{0, 1, 2, 3, 4, 5, 6}};

    REQUIRE(v.size() == int3{3, 2, 7});

    v.resize(5);
    REQUIRE(v.size() == int3{5, 5, 5});

    v.resize(int3{1, 2, 3});
    REQUIRE(v.size() == int3{1, 2, 3});
}

TEST_CASE("from scalar")
{
    using namespace ccs;

    auto x =
        scalar_field<real, 0>{std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, int3{3, 2, 1}};

    vector_field v =
        scalar_field<real, 0>{std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, int3{3, 2, 1}};

    auto vz = v.z;
    vz += v.y;
    vz += v.x;

    scalar_field<real, 0> xz = vz;

    REQUIRE(rs::equal(xz, x + x + x));
}

TEST_CASE("selection")
{
    using namespace ccs;

    const int3 ext = int3{3, 2, 1};
    auto x = scalar_field<real, 0>{std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, ext};

    using I = std::vector<int3>;
    using T = std::vector<real>;

    SECTION("vector selector on scalar")
    {
        // As a convenience allow for a vector range selector from a scalar field
        // to produce another vector range
        vector_range<T> r{};
        r <<= x >> select(vector_range{
                       I{{0, 0, 0}, {0, 1, 0}}, I{{1, 0, 0}, {2, 0, 0}}, I{{2, 1, 0}}});

        REQUIRE(r.size() == int3{2, 2, 1});
        REQUIRE(r.x == T{1, 4});
        REQUIRE(r.y == T{2, 3});
        REQUIRE(r.z == T{6});
    }

    SECTION("scalar selector on vector")
    {
        vector_field v = scalar_field<real, 0>{T{1, 2, 3, 4, 5, 6}, ext};

        v >> select(I{{2, 1, 0}, {0, 0, 0}}) <<= T{-6, -1};
        REQUIRE(rs::equal(v.x, T{-1, 2, 3, 4, 5, -6}));
        REQUIRE(rs::equal(x_field{v.y}, T{-1, 2, 3, 4, 5, -6}));
        REQUIRE(rs::equal(x_field{v.z}, T{-1, 2, 3, 4, 5, -6}));

        v >> select(I{{2, 1, 0}, {0, 0, 0}}) <<= vector_range{T{6, 1}, T{-12, -2}, T{-8}};
        REQUIRE(rs::equal(v.x, T{1, 2, 3, 4, 5, 6}));
        REQUIRE(rs::equal(x_field{v.y}, T{-2, 2, 3, 4, 5, -12}));
        REQUIRE(rs::equal(x_field{v.z}, T{-1, 2, 3, 4, 5, -8}));
    }

    SECTION("vector selector on vector")
    {
        vector_field v = scalar_field<real, 0>{T{1, 2, 3, 4, 5, 6}, ext};
        vector_range<T> r{};

        auto sel = select(vector_range{
            I{{0, 1, 0}}, I{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}}, I{{2, 1, 0}, {2, 0, 0}}});

        v >> sel <<= T{10, 11, 12};
        REQUIRE(rs::equal(v.x, T{1, 2, 3, 10, 5, 6}));
        REQUIRE(rs::equal(x_field{v.y}, T{10, 11, 12, 4, 5, 6}));
        REQUIRE(rs::equal(x_field{v.z}, T{1, 2, 11, 4, 5, 10}));
        r <<= v >> sel;
        REQUIRE(r.x == T{10});
        REQUIRE(r.y == T{10, 11, 12});
        REQUIRE(r.z == T{10, 11});

        v >> sel <<= vector_range{T{0}, T{-1, -2, -3}, T{-4, -5}};
        REQUIRE(rs::equal(v.x, T{1, 2, 3, 0, 5, 6}));
        REQUIRE(rs::equal(x_field{v.y}, T{-1, -2, -3, 4, 5, 6}));
        REQUIRE(rs::equal(x_field{v.z}, T{1, 2, -5, 4, 5, -4}));
        r <<= v >> sel;
        REQUIRE(r.x == T{0});
        REQUIRE(r.y == T{-1, -2, -3});
        REQUIRE(r.z == T{-4, -5});

    }
}