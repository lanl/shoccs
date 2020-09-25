#include "gradient.hpp"
#include "identity_stencil.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "random/random.hpp"

#include <range/v3/algorithm/equal.hpp>

TEST_CASE("domain")
{
    using namespace ccs;

    stencil st{identity_stencil{}};
    const int3 extents{12, 10, 13};
    auto m = mesh{real3{}, real3{1, 1, 1}, extents};

    auto dd = domain_boundaries{boundary::dirichlet, boundary::dirichlet};
    auto nn = domain_boundaries{boundary::neumann, boundary::neumann};
    grid_boundaries grid_b{dd, nn, dd};
    object_boundaries obj_b{};

    auto g = geometry{};

    auto grad = op::gradient{st, m, g, grid_b, obj_b};

    randomize();

    auto solution =
        vs::generate_n([]() { return pick(); }, m.size()) | rs::to<std::vector<real>>();

    x_field f{solution, extents};
    x_field df{solution, extents};

    // no objects in domain
    vector_field_bvalues f_bvals{};
    vector_field_bvalues df_bvals{};

    vector_field<real> dxyz{extents};

    grad(f, df, f_bvals, df_bvals, dxyz);
    // check x-direction results
    REQUIRE(rs::equal(f, dxyz.x()));
    // check y-direction results
    y_field yf{f};
    REQUIRE(rs::equal(yf, dxyz.y()));
    // check z-direction results
    z_field zf{yf};
    REQUIRE(rs::equal(zf, dxyz.z()));
}

TEST_CASE("objects")
{
    using namespace ccs;

    stencil st{identity_stencil{}};
    const int3 extents{22, 25, 32};
    auto m = mesh{real3{-1, -1, -1}, real3{1, 1, 1}, extents};

    auto shapes = std::vector{make_sphere(0, real3{-0.1, -0.2, -0.3}, 0.5),
                              make_sphere(1, real3{0.2, 0.3, 0.4}, 0.49)};

    auto dd = domain_boundaries{boundary::dirichlet, boundary::dirichlet};
    auto nn = domain_boundaries{boundary::neumann, boundary::neumann};
    grid_boundaries grid_b{dd, nn, dd};
    object_boundaries obj_b{boundary::dirichlet, boundary::neumann};

    auto geom = geometry{shapes, m};

    auto grad = op::gradient{st, m, g, grid_b, obj_b};

    randomize();

    auto solution =
        vs::generate_n([]() { return pick(); }, m.size()) | rs::to<std::vector<real>>();

    x_field f{solution, extents};
    x_field df{solution, extents};
    vector_field<real> dxyz{x_field{solution, extents}};

    // no objects in domain
    vector_field_bvalues f_bvals{};
    vector_field_bvalues df_bvals{};

    vector_field<real> dxyz{extents};

    grad(f, df, f_bvals, df_bvals, dxyz);
    // check x-direction results
    REQUIRE(rs::equal(f, dxyz.x()));
    // check y-direction results
    y_field yf{f};
    REQUIRE(rs::equal(yf, dxyz.y()));
    // check z-direction results
    z_field zf{yf};
    REQUIRE(rs::equal(zf, dxyz.z()));
}