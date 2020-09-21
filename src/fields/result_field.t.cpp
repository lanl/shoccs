#include "result_field.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <range/v3/algorithm/equal.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/repeat_n.hpp>

#include <vector>
#include <span>

TEST_CASE("construction") {
    
    using namespace ccs;

    namespace rs = ranges;

    auto x = std::vector<real>{1, 2, 3};
    auto y = result_view(x);

    y += result_range(vs::iota(1));

    REQUIRE(rs::equal(x, std::vector{2, 4, 6}));

}