#include "result_field_range.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <range/v3/algorithm/equal.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/repeat_n.hpp>
#include <range/v3/view/drop.hpp>
#include <range/v3/view/concat.hpp>

#include <vector>
#include <span>

TEST_CASE("construction") {
    
    using namespace ccs;

    namespace rs = ranges;

    auto x_ = std::vector{1, 2, 3, 4, 5, 6};
    auto y_ = std::vector{-1, -2, -3, -4, -5, -6};
    auto x = result_range(x_ | vs::all);
    //field<std::vector, real>{std::vector<real>{1, 2, 3, 4, 5, 6}};
    //auto y_ = std::vector<real>{-1, -2, -3, -4, -5, -6};
    auto y = result_range(y_ | vs::all); //field<std::span, real>{y_};

    auto sum = x + y + 2 * x - y + 3 * y;

    REQUIRE(rs::equal(sum, vs::repeat_n(0.0, ranges::size(sum))));

    REQUIRE(rs::contiguous_range<decltype(x)>);
    REQUIRE(rs::contiguous_range<decltype(y)>);
    REQUIRE(!rs::contiguous_range<decltype(sum)>);



}