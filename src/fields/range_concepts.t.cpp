#include "types.hpp"

#include <catch2/catch_test_macros.hpp>

#include <range/v3/range/concepts.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/algorithm/equal.hpp>

#include <vector>

namespace ccs
{
template <typename T>
concept any_output_range = rs::range<T>&& rs::output_range<T, rs::range_value_t<T>>;
}

TEST_CASE("Output Ranges")
{
    using namespace ccs;

    REQUIRE(rs::output_range<std::vector<real>&, real>);
    REQUIRE(any_output_range<std::vector<real>&>);

    REQUIRE(!rs::output_range<const std::vector<real>&, real>);
    REQUIRE(!any_output_range<const std::vector<real>&>);

    REQUIRE(any_output_range<std::span<real>>);
    REQUIRE(!any_output_range<std::span<const real>>);
}

TEST_CASE("Modify Containers from Views")
{
    using namespace ccs;

    auto x = std::vector{1, 2, 3};
    auto y = vs::all(x);

    REQUIRE(any_output_range<decltype(y)>);

    for (auto&& i : y) i *= 2;

    REQUIRE(rs::equal(x, std::vector{2, 4, 6}));
}