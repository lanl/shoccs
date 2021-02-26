#include "derivative.hpp"
#include "identity_stencil.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>


#include "random/random.hpp"
#include "fields/matchers.hpp"

#include <range/v3/algorithm/equal.hpp>



TEST_CASE("identity")
{
    using namespace ccs;
    using Catch::Matchers::Approx;

    stencil st{identity_stencil{}};
    const int3 extents{12, 10, 13};
    auto m = mesh{real3{}, real3{1, 1, 1}, extents};

    auto dd = domain_boundaries{boundary::dirichlet, boundary::dirichlet};
    auto nn = domain_boundaries{boundary::neumann, boundary::neumann};
    grid_boundaries grid_b{dd, nn, dd};
    object_boundaries obj_b{};

    auto g = geometry{};

    auto deriv = op::derivative{op::directional{0, st, m, g, grid_b[0], obj_b},
                                op::directional{1, st, m, g, grid_b[1], obj_b},
                                op::directional{2, st, m, g, grid_b[2], obj_b}};

    randomize();

    auto solution =
        vs::generate_n([]() { return pick(); }, m.size()) | rs::to<std::vector<real>>();

    vector_field<real> f {x_field{solution, extents}};
    vector_field<real> df {x_field{solution, extents}};

    vector_field<real> dxyz{extents};

    deriv(f, df, dxyz);
    
    REQUIRE_THAT(f, Approx(dxyz));
}