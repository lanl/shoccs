#include "gradient.hpp"
#include "identity_stencil.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "random/random.hpp"

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

    auto solution = vs::generate_n(pick, m.size()) | rs::to<std::vector<real>>();
}