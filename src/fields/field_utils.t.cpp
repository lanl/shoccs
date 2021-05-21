#include "field_utils.hpp"
#include "field.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <range/v3/all.hpp>

using namespace ccs;
using T = std::vector<real>;

TEST_CASE("for_each")
{
    auto x = field{system_size{1, 0, tuple{tuple{5}, tuple{1, 2, 3}}}};

    for_each_scalar([](auto& v) { v += 1; }, x);

    {
        auto&& [xs] = x.scalars(0);
        REQUIRE(rs::equal(xs | sel::D, vs::repeat_n(1, 5)));
        REQUIRE(rs::equal(xs | sel::Rx, vs::repeat_n(1, 1)));
        REQUIRE(rs::equal(xs | sel::Ry, vs::repeat_n(1, 2)));
        REQUIRE(rs::equal(xs | sel::Rz, vs::repeat_n(1, 3)));
    }

    auto y = field{system_size{1, 0, tuple{tuple{5}, tuple{1, 2, 3}}}};

    for_each_scalar(
        [](auto& u, auto& v) {
            v += 1;
            u += v;
        },
        x,
        y);

    {
        auto&& [xs] = x.scalars(0);
        REQUIRE(rs::equal(xs | sel::D, vs::repeat_n(2, 5)));
        REQUIRE(rs::equal(xs | sel::Rx, vs::repeat_n(2, 1)));
        REQUIRE(rs::equal(xs | sel::Ry, vs::repeat_n(2, 2)));
        REQUIRE(rs::equal(xs | sel::Rz, vs::repeat_n(2, 3)));
    }
}
