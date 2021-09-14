#include "unit_stride_visitor.hpp"
#include "circulant.hpp"
#include "csr.hpp"
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

TEST_CASE("no boundary with holes")
{
    auto t = vs::repeat(0.0);

    auto vis = matrix::unit_stride_visitor(10, 10);

    {
        const auto mat = matrix::dense{2, 3, 1, 0, 1, t};
        mat.visit(vis);
    }

    REQUIRE(rs::equal(vis.mapped(1, 2, 0, 3), std::vector{0, 1, 2, 3, 4, 5}));

    {
        const auto mat = matrix::dense{3, 4, 7, 6, 1, t};
        mat.visit(vis);
    }

    REQUIRE(rs::equal(vis.mapped(1, 2, 0, 3), std::vector{0, 1, 2, 7, 8, 9}));
    REQUIRE(rs::equal(vis.mapped(7, 3, 6, 4),
                      std::vector{17, 18, 19, 20, 24, 25, 26, 27, 31, 32, 33, 34}));
}

TEST_CASE("dirichlet")
{

    auto t = vs::repeat(0.0);

    auto vis = matrix::unit_stride_visitor(10, 10);

    {
        const auto mat = matrix::dense{2, 3, 1, 0, 1, t, ldd};
        mat.visit(vis);
    }

    REQUIRE(rs::equal(vis.mapped(1, 2, 1, 2), std::vector{0, 1, 2, 3}));

    {
        const auto mat = matrix::dense{2, 4, 7, 6, 1, t, rdd};
        mat.visit(vis);
    }

    REQUIRE(rs::equal(vis.mapped(1, 2, 1, 2), std::vector{0, 1, 5, 6}));
    REQUIRE(rs::equal(vis.mapped(7, 2, 6, 3), std::vector{12, 13, 14, 17, 18, 19}));
}

TEST_CASE("inner_block")
{
    auto t = T(3, 0);
    auto u = vs::repeat(0.0);

    auto vis = matrix::unit_stride_visitor(10, 10);

    auto x = matrix::inner_block(10,
                                 1,
                                 0,
                                 1,
                                 matrix::dense{2, 3, u, ldd},
                                 matrix::circulant(5, t),
                                 matrix::dense{2, 3, u});

    x.visit(vis);

    REQUIRE(rs::equal(vis.mapped(1, 2, 1, 2), std::vector{0, 1, 9, 10}));
    REQUIRE(rs::equal(vis.mapped(3, 1, 2, 3), std::vector{19, 20, 21}));
    REQUIRE(rs::equal(vis.mapped(8, 2, 7, 3), std::vector{69, 70, 71, 78, 79, 80}));
}

TEST_CASE("csr")
{
    int3 nxyz{10, 1, 1};
    // support an objects on the left/right with dirichlet/floating bcs, resp.
    auto vis = matrix::unit_stride_visitor(
        nxyz, std::vector<bool>{true, false}, std::vector<bool>{}, std::vector<bool>{});

    auto t = T(3, 0);
    auto u = vs::repeat(0.0);

    auto x = matrix::inner_block(8,
                                 1,
                                 1,
                                 1,
                                 matrix::dense{2, 2, u},
                                 matrix::circulant(4, t),
                                 matrix::dense{2, 2, u});

    x.visit(vis);

    // B: rowspace: f; colspace rx
    auto B_ = matrix::csr::builder();
    // left dirichlet boundary
    B_.add_point(1, 0, 0.);
    B_.add_point(2, 0, 0.);
    // right floating boundary
    B_.add_point(7, 1, 0.);
    B_.add_point(8, 1, 0.);

    auto B = B_.to_csr(nxyz[0]);
    B.flags(colspace_rx);

    // Bfx: rowspace: rx; colspace f
    auto Bf_ = matrix::csr::builder();
    // right floating boundary
    Bf_.add_point(1, 7, 0.);
    Bf_.add_point(1, 8, 0.);

    auto Bf = Bf_.to_csr(2);
    Bf.flags(rowspace_rx);

    // Brx: rowspace: rx; colspace rx
    auto Br_ = matrix::csr::builder();
    Br_.add_point(1, 1, 0.);

    auto Br = Br_.to_csr(2);
    Br.flags(rowspace_rx | colspace_rx);

    B.visit(vis);
    Bf.visit(vis);
    Br.visit(vis);

    REQUIRE(vis.mapped_size() == 9 * 9);
}
