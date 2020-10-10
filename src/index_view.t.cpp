#include "index_view.hpp"

#include <catch2/catch_test_macros.hpp>

#include <range/v3/algorithm/equal.hpp>
#include <range/v3/range/conversion.hpp>
#include <vector>

TEST_CASE("plane_view")
{
    using namespace ccs;
    using T = int3;
    int3 extents = {3, 2, 4};

    auto x = index_view<0>(extents, 0) | rs::to<std::vector<int3>>();
    REQUIRE(x == std::vector<int3>{T{0, 0, 0},
                                   T{0, 0, 1},
                                   T{0, 0, 2},
                                   T{0, 0, 3},
                                   T{0, 1, 0},
                                   T{0, 1, 1},
                                   T{0, 1, 2},
                                   T{0, 1, 3}});
    REQUIRE(rs::equal(index_view<0>(extents, -1),
                      std::vector<int3>{T{2, 0, 0},
                                        T{2, 0, 1},
                                        T{2, 0, 2},
                                        T{2, 0, 3},
                                        T{2, 1, 0},
                                        T{2, 1, 1},
                                        T{2, 1, 2},
                                        T{2, 1, 3}}));

    REQUIRE(rs::equal(index_view<1>(extents, 0),
                      std::vector<int3>{T{0, 0, 0},
                                        T{0, 0, 1},
                                        T{0, 0, 2},
                                        T{0, 0, 3},
                                        T{1, 0, 0},
                                        T{1, 0, 1},
                                        T{1, 0, 2},
                                        T{1, 0, 3},
                                        T{2, 0, 0},
                                        T{2, 0, 1},
                                        T{2, 0, 2},
                                        T{2, 0, 3}}));
    REQUIRE(rs::equal(index_view<1>(extents, -1),
                      std::vector<int3>{T{0, 1, 0},
                                        T{0, 1, 1},
                                        T{0, 1, 2},
                                        T{0, 1, 3},
                                        T{1, 1, 0},
                                        T{1, 1, 1},
                                        T{1, 1, 2},
                                        T{1, 1, 3},
                                        T{2, 1, 0},
                                        T{2, 1, 1},
                                        T{2, 1, 2},
                                        T{2, 1, 3}}));

    REQUIRE(rs::equal(
        index_view<2>(extents, 0),
        std::vector<int3>{
            T{0, 0, 0}, T{0, 1, 0}, T{1, 0, 0}, T{1, 1, 0}, T{2, 0, 0}, T{2, 1, 0}}));
    REQUIRE(rs::equal(
        index_view<2>(extents, -1),
        std::vector<int3>{
            T{0, 0, 3}, T{0, 1, 3}, T{1, 0, 3}, T{1, 1, 3}, T{2, 0, 3}, T{2, 1, 3}}));
}