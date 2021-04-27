#include "DiscreteOperator.hpp"
#include "Gradient.hpp"

#include "IdentityStencil.hpp"

#include "mesh/Mesh.hpp"

#include "fields/Selector.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "random/random.hpp"

#include <range/v3/algorithm/count.hpp>
#include <range/v3/algorithm/equal.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/transform.hpp>

TEST_CASE("domain")
{
    using namespace ccs;
    using namespace ccs::stencils;
    // using Catch::Matchers::Approx;

    Stencil stencil = Identity;
    REQUIRE(stencil);
    // const int3 extents{31, 33, 32};
    // auto m = mesh::Cartesian{real3{-1, -1, -1}, real3{1, 1, 1}, extents};
    // auto g = mesh::CutGeometry{};

    auto m = mesh::Mesh{mesh::IndexExtents{31, 33, 32},
                        mesh::DomainBounds{real3{-1, -1, -1}, real3{1, 1, 1}}};

    auto gradient = operators::DiscreteOperator(Identity, m)
                        .to<operators::Gradient>(bcs::Grid{bcs::dd, bcs::ff, bcs::dd});

    randomize();

#if 0
    auto u = field::Scalar<std::vector<real>>{m};
    u | selector::D = mesh::location | vs::transform([](auto&&...) { return pick(); });
    u | selector::Rxyz = 0;

    auto du = field::Vector<std::vector<real>>{m};

    du = gradient(u);


    auto dd = domain_boundaries{boundary::dirichlet, boundary::dirichlet};
    auto nn = domain_boundaries{boundary::neumann, boundary::neumann};
    grid_boundaries grid_b{dd, nn, dd};
    object_boundaries obj_b{};


    auto grad = op::gradient{st, m, g, grid_b, obj_b};

    randomize();

    auto solution =
        vs::generate_n([]() { return pick(); }, m.size()) | rs::to<std::vector<real>>();

    const x_field f{solution, extents};
    const x_field df{solution, extents};

    // no objects in domain
    vector_range<std::vector<real>> f_bvals{};
    vector_range<std::vector<real>> df_bvals{};

    vector_field<real> dxyz{extents};

    grad(f, df, f_bvals, df_bvals, dxyz);

    vector_field<real> vf{f};
    REQUIRE_THAT(vf, Approx(dxyz));
#endif
}