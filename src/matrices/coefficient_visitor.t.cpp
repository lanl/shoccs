#include "coefficient_visitor.hpp"
#include "circulant.hpp"
#include "dense.hpp"
#include "inner_block.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <range/v3/all.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

using namespace ccs;
using Catch::Matchers::Approx;
using T = std::vector<real>;

TEST_CASE("simple dense")
{
    T imat = vs::iota(25, 50) | rs::to<T>();

    const auto mat = matrix::dense{5, 5, imat};

    auto v = matrix::unit_stride_visitor(5, 5);
    mat.visit(v);

    auto u = matrix::coefficient_visitor(MOVE(v));

    mat.visit(u);

    REQUIRE(rs::equal(imat, u.matrix()));
}

TEST_CASE("inner block")
{

    T imat = vs::iota(25, 50) | rs::to<T>();
    T cmat{1, -2, 1};

    auto x = matrix::inner_block{11,
                                 1,
                                 0,
                                 1,
                                 matrix::dense{2, 3, imat, ldd},
                                 matrix::circulant{5, cmat},
                                 matrix::dense{2, 4, imat, rdd}};

    auto v = matrix::unit_stride_visitor(11, 11);
    x.visit(v);

    auto u = matrix::coefficient_visitor(MOVE(v));
    x.visit(u);

    REQUIRE(u.matrix().size() == 81u);

    T expected{26, 27, 0,  0,  0,  0,  0,  0,  0,  /* row 0 */
               29, 30, 0,  0,  0,  0,  0,  0,  0,  /* row 1 */
               0,  1,  -2, 1,  0,  0,  0,  0,  0,  /* row 2 */
               0,  0,  1,  -2, 1,  0,  0,  0,  0,  /* row 3 */
               0,  0,  0,  1,  -2, 1,  0,  0,  0,  /* row 4 */
               0,  0,  0,  0,  1,  -2, 1,  0,  0,  /* row 5 */
               0,  0,  0,  0,  0,  1,  -2, 1,  0,  /* row 6 */
               0,  0,  0,  0,  0,  0,  25, 26, 27, /* row 7 */
               0,  0,  0,  0,  0,  0,  29, 30, 31};
    REQUIRE(rs::equal(expected, u.matrix()));
}
