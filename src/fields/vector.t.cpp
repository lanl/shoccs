#include "types.hpp"

#include "Vector.hpp"

#include "Selector.hpp"

#include "lift.hpp"
#include "views.hpp"

#include <catch2/catch_test_macros.hpp>
#include <range/v3/algorithm/equal.hpp>

// construct simple mesh geometry
namespace global
{

const auto x = std::vector<ccs::real>{0, 1, 2, 3, 4};
const auto y = std::vector<ccs::real>{-2, -1};
const auto z = std::vector<ccs::real>{6, 7, 8, 9};
const auto rx = std::vector<ccs::real3>{{0.5, -2, 6}, {1.5, -1, 9}};
const auto ry = std::vector<ccs::real3>{{1, -1.75, 7}, {4, -1.25, 7}, {3, -1.1, 9}};
const auto rz = std::vector<ccs::real3>{{0, -2, 6.1}};
const auto loc = ccs::mesh::Location{x, y, z, rx, ry, rz};

} // namespace global

TEST_CASE("construction")
{
    using namespace ccs::field;
    using T = std::vector<int>;
    auto s = SimpleVector<T>{};
}