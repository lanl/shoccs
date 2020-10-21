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
    auto s = x_scalar{x_field{vs::iota(0), int3{4, 5, 6}},
                      vector_range{T{0, 1, 2}, T{4, 5}, T{6, 7, 8}},
                      vector_range<span<const int3>>{}};
    auto s1 = x_scalar{x_field{vs::iota(1), int3{4, 5, 6}},
                       vector_range{T{1, 2, 3}, T{5, 6}, T{7, 8, 9}},
                       vector_range<span<const int3>>{}};

    s += 1;
    REQUIRE_THAT(s, Approx(s1));

    s = 0;
    REQUIRE_THAT(s, Approx(x_scalar{int3{4, 5, 6}, int3{3, 2, 3}}));

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

    // some intialization
    auto s = x_scalar{
        x_field{vs::iota(0), int3{4, 5, 6}},
        vector_range{T{0, 1, 2}, T{4, 5}, T{6, 7, 8}},
        vector_range<std::span<const int3>>{v_arg{xpts}, v_arg{ypts}, v_arg{zpts}}};

    // need to size bcs since the index_view is a generator without a size.
    std::vector<real> bcs(30);
    bcs <<= s.field >> select(index_view<0>(s.extents(), 0));
    REQUIRE(rs::equal(bcs, vs::iota(0) | vs::stride(4) | vs::take(30)));

    bcs <<= s >> field_select(index_view<0>(s.extents(), -1));
    REQUIRE(rs::equal(bcs, vs::iota(3) | vs::stride(4) | vs::take(30)));

    // check assignment
    s >> field_select(index_view<0>(s.extents(), 0)) <<= bcs;
    auto x_max_bcs = bcs;
    bcs <<= s >> field_select(index_view<0>(s.extents(), 0));
    REQUIRE(x_max_bcs == bcs);

    s >> obj_select() <<= -1;
    REQUIRE(s.obj.xi(2) == -1);
    REQUIRE(s.obj.zi(0) == -1);
    REQUIRE(s.field(xpts[0]) == -1);

    std::vector<real> pts{};
    pts <<= s >> field_select(xpts);

    REQUIRE(pts.size() == xpts.size());
    REQUIRE(pts == std::vector<real>(pts.size(), -1));
}