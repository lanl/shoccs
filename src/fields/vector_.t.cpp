#include "index_view.hpp"
#include "scalar.hpp"
#include "select.hpp"

#include "matchers.hpp"
#include <catch2/catch_test_macros.hpp>

#include <range/v3/algorithm/equal.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/stride.hpp>
#include <range/v3/view/take.hpp>

TEST_CASE("math")
{
    using namespace ccs;
    using Catch::Matchers::Approx;
    using T = std::vector<real>;

    // some intialization
    vec s{v_field{x_field{vs::iota(0), int3{4, 5, 6}}},
          vector_range{T{0, 1, 2}, T{4, 5}, T{6, 7, 8}},
          vector_range<span<const int3>>{}};
    vec s1{v_field{x_field{vs::iota(1), int3{4, 5, 6}}},
           vector_range{T{1, 2, 3}, T{5, 6}, T{7, 8, 9}},
           vector_range<span<const int3>>{}};

    s += 1;
    REQUIRE_THAT(s, Approx(s1));

    s = 0;
    s += s1 * 1;
    REQUIRE_THAT(s, Approx(s1));

    s = s1 * 2 - s1;
    REQUIRE_THAT(s, Approx(s1));

    s += 3 * s1 - 2 * s1 - s1;
    REQUIRE_THAT(s, Approx(s1));
}

TEST_CASE("selection")
{
    using namespace ccs;
    using Catch::Matchers::Approx;
    using T = std::vector<real>;

    auto xpts = std::vector<int3>{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    auto ypts = std::vector<int3>{{0, 3, 0}, {0, 4, 0}, {1, 0, 0}};
    auto zpts = std::vector<int3>{{3, 4, 5}, {2, 2, 2}};

    vec s{v_field{x_field{vs::iota(0), int3{4, 5, 6}}},
          vector_range{T{0, 1, 2}, T{4, 5, 6}, T{7, 8}},
          vector_range<span<const int3>>{v_arg{xpts}, v_arg{ypts}, v_arg{zpts}}};

    std::vector<real> bcs(30);
    bcs <<= s >> x_field_select(index_view<0>(s.extents(), 0));
    REQUIRE(rs::equal(bcs, vs::iota(0) | vs::stride(4) | vs::take(30)));

    // grab the first two points in each field
    auto r = vector_range<T>{};
    r <<= s >> field_select(vs::iota(0, 2));
    REQUIRE_THAT(r, Approx(vector_range<T>{{0, 1}, {0, 24}, {0, 4}}));

    s >> obj_select() <<= -1;
    r <<= s >> field_select(s.m);

    REQUIRE(r.size() == int3{3, 3, 2});
    REQUIRE(r.x == T(3, -1));
    REQUIRE(r.y == T(3, -1));
    REQUIRE(r.z == T(2, -1));
    //REQUIRE_THAT(r, Approx(vector_range<T>{T(3, -1), T(3, -1), T(2, -1)}));
}