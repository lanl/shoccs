#include "Tuple.hpp"

#include "types.hpp"

#include <catch2/catch_test_macros.hpp>

#include <range/v3/algorithm/equal.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/zip_with.hpp>

#include <vector>

TEST_CASE("container math")
{
    using namespace ccs;
    using namespace ccs::field::tuple;
    using T = std::vector<real>;

    auto x = Tuple<T>{vs::iota(0, 10)};
    x += 5;
    REQUIRE(rs::equal(x, vs::iota(5, 15)));

    Tuple<T> y = Tuple{vs::iota(0, 10)};
    y += 5;
    REQUIRE(rs::equal(x, y));

    auto xx = Tuple<T, T>{vs::iota(-5, 3), vs::iota(-10, 1)};
    auto yy = Tuple{vs::iota(-5, 3), vs::iota(-10, 1)};
    xx += 100;
    REQUIRE(rs::equal(view<0>(xx), vs::iota(95, 103)));
    REQUIRE(rs::equal(view<1>(xx), vs::iota(90, 101)));

    xx += yy;
    REQUIRE(rs::equal(view<0>(xx),
                      vs::zip_with(std::plus{}, vs::iota(95, 103), vs::iota(-5, 3))));
    REQUIRE(rs::equal(view<1>(xx),
                      vs::zip_with(std::plus{}, vs::iota(90, 101), vs::iota(-10, 1))));
}

TEST_CASE("view math with numeric")
{
    using namespace ccs;
    using namespace ccs::field::tuple;

    using T = std::vector<real>;

    for (int i = 100; i < 1000; i += 10) {
        {
            auto x = Tuple<T>{vs::iota(-i, i)};
            auto z = 5 + x + 5;

            REQUIRE(rs::equal(z, vs::iota(10 - i, 10 + i)));
            REQUIRE(rs::equal(z, vs::iota(10 - i, 10 + i)));
        }

        {
            auto vx = vs::iota(-i, 0);
            auto vy = vs::iota(0, i);
            auto vz = vs::iota(-1, 2 * i);
            auto xx = Tuple<T, T, T>{vx, vy, vz};
            auto zz = (xx + 5) + 1;

            auto o = vs::repeat(6);

            REQUIRE(rs::equal(view<0>(zz), vs::zip_with(std::plus{}, vx, o)));
            REQUIRE(rs::equal(view<1>(zz), vs::zip_with(std::plus{}, vy, o)));
            REQUIRE(rs::equal(view<2>(zz), vs::zip_with(std::plus{}, vz, o)));
        }

        {
            auto x = vs::iota(0, i);
            auto y = vs::iota(i, 2 * i);
            auto z = vs::iota(2 * i, 3 * i);
            auto xyz = Tuple<T, T, T>(x, y, z);

            auto s = xyz + Tuple{x};

            REQUIRE(rs::equal(view<0>(s), vs::zip_with(std::plus{}, x, x)));
            REQUIRE(rs::equal(view<1>(s), vs::zip_with(std::plus{}, y, x)));
            REQUIRE(rs::equal(view<2>(s), vs::zip_with(std::plus{}, z, x)));
        }
    }
}

TEST_CASE("view math with tuples")
{
    using namespace ccs;
    using namespace ccs::field::tuple;

    using T = std::vector<real>;

    auto x = Tuple<T>{vs::iota(0, 10)};
    auto y = Tuple{vs::iota(10, 20)};
    auto z = x + y;

    REQUIRE(rs::equal(z, vs::zip_with(std::plus{}, vs::iota(0, 10), vs::iota(10, 20))));

    auto g = z + x + x + y;
    auto h = z + z + x;

    REQUIRE(g.size() == h.size());
    REQUIRE(rs::equal(g, h));

    {
        auto xx = Tuple<T>{x + y};
        REQUIRE(rs::equal(z, xx));

        auto yy = Tuple<T>{};
        yy = x + y;
        REQUIRE(rs::equal(yy, xx));
    }

    {
        auto xx = Tuple<T, T, T>{x + y, z + x + x + y, z + z + x};
        REQUIRE(rs::equal(view<0>(xx), z));
        REQUIRE(rs::equal(view<1>(xx), g));
        REQUIRE(rs::equal(view<2>(xx), h));

        auto yy = Tuple<T, T, T>{};
        yy = Tuple{x + y, z + x + x + y, z + z + x};
        REQUIRE(rs::equal(view<0>(yy), z));
        REQUIRE(rs::equal(view<1>(yy), g));
        REQUIRE(rs::equal(view<2>(yy), h));
    }
}