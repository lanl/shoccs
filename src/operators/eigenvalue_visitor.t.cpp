#include "eigenvalue_visitor.hpp"
#include "derivative.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "fields/tuple_utils.hpp"
#include "identity_stencil.hpp"
#include "stencils/stencil.hpp"

#include <sol/sol.hpp>

using namespace ccs;
using Catch::Matchers::Approx;
using T = std::vector<real>;
using B = std::vector<bool>;

TEST_CASE("identity")
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(            
            simulation = {
                mesh = {
                    index_extents = {11},
                    domain_bounds = {1}
                },
                shapes = {
                    {
                        type = "yz_rect",
                        psi = 0.1,
                        normal = 1,
                        boundary_condition = "dirichlet"
                    },
                    {
                        type = "yz_rect",
                        psi = 0.2,
                        normal = -1,
                        boundary_condition = "floating"
                    }
                }
            }
        )");

    auto mesh_opt = mesh::from_lua(lua["simulation"]);
    REQUIRE(!!mesh_opt);

    auto bc_opt = bcs::from_lua(lua["simulation"], mesh_opt->extents());
    REQUIRE(!!bc_opt);

    auto dx = derivative{0, *mesh_opt, stencils::identity, bc_opt->first, bc_opt->second};

    auto v = eigenvalue_visitor{mesh_opt->extents(), B{true, false}, B{}, B{}};
    v.visit(dx);

    auto eigs = to<T>(v.eigenvalues_real());
    REQUIRE(eigs.size() == 10u);

    T exact(eigs.size(), 1.0);
    REQUIRE_THAT(eigs, Approx(exact));
}

TEST_CASE("e2-poly")
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(            
            simulation = {
                mesh = {
                    index_extents = {21},
                    domain_bounds = {20}
                },
                shapes = {
                    {
                        type = "yz_rect",
                        psi = 0.001,
                        normal = 1,
                        boundary_condition = "dirichlet"
                    },
                    {
                        type = "yz_rect",
                        psi = 0.9,
                        normal = -1,
                        boundary_condition = "floating"
                    }
                },
                scheme = {
                    order = 1,
                    type = "E2-poly",
                    floating_alpha = {13/100, 7/50, 3/20, 4/25, 17/100, 9/50},
                    dirichlet_alpha = {3/25, 13/100, 7/50}
                }
            }
        )");

    auto mesh_opt = mesh::from_lua(lua["simulation"]);
    REQUIRE(!!mesh_opt);

    auto bc_opt = bcs::from_lua(lua["simulation"], mesh_opt->extents());
    REQUIRE(!!bc_opt);

    auto st_opt = stencil::from_lua(lua["simulation"]);
    REQUIRE(!!st_opt);

    auto dx = derivative{0, *mesh_opt, *st_opt, bc_opt->first, bc_opt->second};

    auto v = eigenvalue_visitor{mesh_opt->extents(), B{true, false}, B{}, B{}};
    v.visit(dx);

    auto eigs = to<T>(v.eigenvalues_real());
    REQUIRE(eigs.size() == 20u);

    REQUIRE(rs::max(eigs) == Catch::Approx(0.19642194054742698));
}
