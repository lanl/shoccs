#include "scalar_field.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <random>
#include <span>
#include <vector>

#include <range/v3/algorithm/equal.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/generate_n.hpp>

TEST_CASE("traits")
{
    using namespace ccs;

    using V = std::vector<real>;
    REQUIRE(std::same_as<const V&, range_t<V>>);
    REQUIRE(std::same_as<const V&, range_t<const V&>>);
    REQUIRE(!detail::Owning<std::span<real>>);
}

TEST_CASE("construction")
{
    using namespace ccs;
    auto x = scalar_field<real>(std::vector{1.0, 2.0, 3.0});
    auto y = scalar_field<real>(std::vector{4.0, 5.0, 6.0});
    // auto z = scalar_field<real, 0>{std::vector{1.0, 2.0, 3.0}};

    auto sum = x + y + 1;
    //auto ans = std::vector<real>{
    //    1 * 1 + 1 + 2 * 4 + 1, 2 * 2 + 2 + 2 * 5 + 1, 3 * 3 + 3 + 2 * 6 + 1};
    auto ans = std::vector<real>{8 + 1, 10 + 1, 12 + 1};

    REQUIRE(ranges::equal(sum, ans));

    auto z = scalar_field<real>{};
    z = sum;

    //REQUIRE(ranges::equal(z, ans));
    //REQUIRE(ranges::equal(sum, z));

    // this shouldn't compile
    // auto proxy = y + z;
}
