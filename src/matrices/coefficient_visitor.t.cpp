#include "coefficient_visitor.hpp"
#include "circulant.hpp"
#include "csr.hpp"
#include "dense.hpp"
#include "inner_block.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <range/v3/all.hpp>

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

TEST_CASE("csr + inner block")
{

    int3 nxyz{10, 1, 1};
    // support an objects on the left/right with dirichlet/floating bcs, resp.
    auto v = matrix::unit_stride_visitor(
        nxyz, std::vector<bool>{true, false}, std::vector<bool>{}, std::vector<bool>{});

    T imat = vs::iota(25, 50) | rs::to<T>();
    T cmat{1, -2, 1};

    auto x = matrix::inner_block{8,
                                 1,
                                 1,
                                 1,
                                 matrix::dense{2, 2, imat},
                                 matrix::circulant{4, cmat},
                                 matrix::dense{2, 3, imat}};

    x.visit(v);

    // B: rowspace: f; colspace rx
    auto B_ = matrix::csr::builder();
    // left dirichlet boundary
    B_.add_point(1, 0, 10.);
    B_.add_point(2, 0, 20.);
    // right floating boundary
    B_.add_point(7, 1, 30.);
    B_.add_point(8, 1, 40.);

    auto B = B_.to_csr(nxyz[0]);
    B.flags(colspace_rx);

    // Bfx: rowspace: rx; colspace f
    auto Bf_ = matrix::csr::builder();
    // right floating boundary
    Bf_.add_point(1, 7, 50.);
    Bf_.add_point(1, 8, 60.);

    auto Bf = Bf_.to_csr(2);
    Bf.flags(rowspace_rx);

    // Brx: rowspace: rx; colspace rx
    auto Br_ = matrix::csr::builder();
    Br_.add_point(1, 1, 70.);

    auto Br = Br_.to_csr(2);
    Br.flags(rowspace_rx | colspace_rx);

    B.visit(v);
    Bf.visit(v);
    Br.visit(v);

    REQUIRE(v.mapped_size() == 81u);

    auto u = matrix::coefficient_visitor(MOVE(v));
    x.visit(u);
    B.visit(u);
    Bf.visit(u);
    Br.visit(u);

    REQUIRE(u.matrix().size() == 81u);

    T expected{25, 26, 0,  0,  0,  0,  0,  0,  0,  /* row 0 */
               27, 28, 0,  0,  0,  0,  0,  0,  0,  /* row 1 */
               0,  1,  -2, 1,  0,  0,  0,  0,  0,  /* row 2 */
               0,  0,  1,  -2, 1,  0,  0,  0,  0,  /* row 3 */
               0,  0,  0,  1,  -2, 1,  0,  0,  0,  /* row 4 */
               0,  0,  0,  0,  1,  -2, 1,  0,  0,  /* row 5 */
               0,  0,  0,  0,  0,  25, 26, 27, 30, /* row 6 */
               0,  0,  0,  0,  0,  28, 29, 30, 40, /* row 7 */
               0,  0,  0,  0,  0,  0,  50, 60, 70};

    REQUIRE(rs::equal(expected, u.matrix()));
}
