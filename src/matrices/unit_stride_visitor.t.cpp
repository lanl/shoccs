#include "unit_stride_visitor.hpp"
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

TEST_CASE("no boundary no holes")
{
    T imat(25);

    const auto mat = matrix::dense{5, 5, imat};

    auto v = matrix::unit_stride_visitor(5, 5);

    mat.visit(v);

    REQUIRE(rs::equal(v.mapped(0, 5, 0, 5), vs::iota(0ul, imat.size())));
    REQUIRE(rs::equal(v.mapped(4, 1, 0, 5), vs::iota(20, 25)));

    const auto mat2 = matrix::dense{5, 5, 5, 5, 1, imat};
    mat2.visit(v);

    REQUIRE(rs::equal(v.mapped(0, 2, 0, 3), std::vector{0, 1, 2, 10, 11, 12}));

    const auto mat3 = matrix::dense{5, 6, 10, 9, 1, imat};
    mat3.visit(v);

    REQUIRE(rs::equal(v.mapped(1, 2, 1, 3), std::vector{16, 17, 18, 31, 32, 33}));
    REQUIRE(rs::equal(v.mapped(14, 1, 0, 15), vs::iota(15 * 14, 15 * 15)));
}

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
