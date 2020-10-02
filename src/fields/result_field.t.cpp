#include "result_field.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <range/v3/algorithm/equal.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/repeat_n.hpp>
#include <range/v3/view/transform.hpp>

#include <span>
#include <vector>

TEST_CASE("result_api")
{

    using namespace ccs;

    auto x_ = std::vector{1, 2, 3, 4, 5, 6};
    auto y_ = std::vector{-1, -2, -3, -4, -5, -6};
    auto x = result_range(x_ | vs::all);
    auto y = result_range(y_ | vs::all);
    auto x_field = result_field{x_};
    auto y_field = result_view_t<const int>{y_};

    auto sum = x + y_field + 2 * x_field - y + 3 * y;

    REQUIRE(rs::equal(sum, vs::repeat_n(0, rs::size(sum))));
    REQUIRE(rs::equal(sum - 1, vs::repeat_n(-1, rs::size(sum))));

    auto range_traits = []<typename T>(const T& r) {
        return rs::contiguous_range<T> && rs::random_access_range<T>;
    };

    REQUIRE(range_traits(x_field));
    REQUIRE(range_traits(y_field));
    REQUIRE(range_traits(x));
    REQUIRE(range_traits(y));
    REQUIRE(!rs::contiguous_range<decltype(sum)>);
    REQUIRE(rs::random_access_range<decltype(sum)>);

    {
        result_field<std::vector, real> xx{};        
        auto t = x >> vs::transform([](auto&& v) { return -v; }) >> vs::transform([](auto&& v) { return v; });
        xx = t;
        REQUIRE(rs::equal(xx, y));
        REQUIRE(Result<decltype(t)>);

        auto tt = t | vs::transform([](auto&& v) {return v; });
        REQUIRE(!Result<decltype(tt)>);
    }

    {
        result_field<std::vector, real> xx = x | vs::transform([](auto&& v) { return -v; });
        REQUIRE(rs::equal(xx, y));
    }
}