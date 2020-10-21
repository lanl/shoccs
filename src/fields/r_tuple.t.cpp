#include "r_tuple.hpp"

#include "types.hpp"

#include <catch2/catch_test_macros.hpp>

#include <range/v3/algorithm/equal.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/take.hpp>

#include <vector>

TEST_CASE("construction")
{
    using namespace ccs;

    auto x = std::vector<real>{0, 1, 2};
    auto y = std::vector<real>{3, 4, 5};
    auto z = std::vector<real>{6, 7, 8};

    auto r = r_tuple(x);
    REQUIRE(std::same_as<decltype(r), r_tuple<std::vector<real>&>>);
    REQUIRE(r[1] == 1);
    r[1] = -1;
    REQUIRE(r[1] == -1);

    REQUIRE(rs::equal(get<0>(r), x));

    SECTION("owning from input range")
    {
        // try different sizes to flush out memory issues
        for (int i = 10; i < 1024; i += 9) {
            auto xx = r_tuple<std::vector<real>>{vs::iota(0, i)};
            REQUIRE(rs::equal(vs::iota(0, i), xx));

            xx = vs::iota(0, i / 2);
            REQUIRE(rs::equal(xx, vs::iota(0, i / 2)));
        }
    }

    SECTION("owning from non-owning")
    {
        auto r_owning = r_tuple{std::vector<real>{0, 1, 2}};
        REQUIRE(r_owning[1] == 1);
        r_owning[1] = -1;
        REQUIRE(r_owning[1] == -1);
        REQUIRE(rs::equal(get<0>(r_owning), x));

        r_owning = r;
        REQUIRE(rs::equal(r_owning, x));
        REQUIRE(rs::equal(r_owning, r));
        auto x0 = x[0];
        // this shouldn't change r/x
        r_owning[0] = 1000;
        REQUIRE(x[0] == x0);
        REQUIRE(r[0] == x0);
        REQUIRE(x0 != 1000);
    }

    auto rr = r_tuple(x, y, z);
    REQUIRE(rr.N == 3);
    REQUIRE(rs::equal(get<0>(rr), x));
    REQUIRE(rs::equal(get<1>(rr), y));
    REQUIRE(rs::equal(get<2>(rr), z));

    {
        auto xx = x;
        auto yy = y;
        auto zz = z;
        auto r_owning = r_tuple{MOVE(xx), MOVE(yy), MOVE(zz)};
        REQUIRE(r_owning.N == 3);
        REQUIRE(rs::equal(get<0>(rr), get<0>(r_owning)));
        REQUIRE(rs::equal(get<1>(rr), get<1>(r_owning)));
        REQUIRE(rs::equal(get<2>(rr), get<2>(r_owning)));
    }
}