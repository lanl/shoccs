#include "indexing.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("transpose")
{
    using namespace ccs;

    int3 ijk = {11, 12, 13};

    int3 ijk_0 = {ijk[0], ijk[index::dir<0>::fast], ijk[index::dir<0>::slow]};
    int3 ijk_1 = {ijk[1], ijk[index::dir<1>::fast], ijk[index::dir<1>::slow]};
    int3 ijk_2 = {ijk[2], ijk[index::dir<2>::fast], ijk[index::dir<2>::slow]};

    REQUIRE(ijk_0 == index::transpose<0, 0>(ijk_0));
    REQUIRE(ijk_0 == index::transpose<1, 0>(ijk_1));
    REQUIRE(ijk_0 == index::transpose<2, 0>(ijk_2));

    REQUIRE(ijk_1 == index::transpose<0, 1>(ijk_0));
    REQUIRE(ijk_1 == index::transpose<1, 1>(ijk_1));
    REQUIRE(ijk_1 == index::transpose<2, 1>(ijk_2));

    REQUIRE(ijk_2 == index::transpose<0, 2>(ijk_0));
    REQUIRE(ijk_2 == index::transpose<1, 2>(ijk_1));
    REQUIRE(ijk_2 == index::transpose<2, 2>(ijk_2));
    
    REQUIRE(ijk_1 == index::transpose<2, 1>(index::transpose<1, 2>(ijk_1)));
}