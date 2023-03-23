#include "laplacian.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <sol/sol.hpp>
#include <spdlog/spdlog.h>

#include "fields/selector.hpp"
#include "identity_stencil.hpp"
#include "random/random.hpp"
#include "stencils/stencil.hpp"

#include <range/v3/all.hpp>

using namespace ccs;
using Catch::Matchers::Approx;

constexpr auto g = []() { return pick(); };

// 2nd order polynomial for use with E2
constexpr auto f2 = vs::transform([](auto&& loc) {
    auto&& [x, y, z] = loc;
    return x * x * (y + z) + y * y * (x + z) + z * z * (x + y) + 3 * x * y * z + x + y +
           z;
});

constexpr auto f2_dx = vs::transform([](auto&& loc) {
    auto&& [x, y, z] = loc;
    return 2. * x * (y + z) + y * y + z * z + 3. * y * z + 1;
});

constexpr auto f2_dy = vs::transform([](auto&& loc) {
    auto&& [x, y, z] = loc;
    return x * x + 2. * y * (x + z) + z * z + 3. * x * z + 1;
});

constexpr auto f2_dz = vs::transform([](auto&& loc) {
    auto&& [x, y, z] = loc;
    return x * x + y * y + 2. * z * (x + y) + 3. * x * y + 1;
});

constexpr auto f2_ddx = vs::transform([](auto&& loc) {
    auto&& [_, y, z] = loc;
    return 2. * (y + z);
});

constexpr auto f2_ddy = vs::transform([](auto&& loc) {
    auto&& [x, _, z] = loc;
    return 2. * (x + z);
});

constexpr auto f2_ddz = vs::transform([](auto&& loc) {
    auto&& [x, y, _] = loc;
    return 2. * (x + y);
});

// 2D 1st order polynomial for use with E2
constexpr auto g2 = vs::transform([](auto&& loc) {
    auto&& [x, y, _] = loc;
    return 3 * x * y + x + y + 1;
});

constexpr auto g2_dx = vs::transform([](auto&& loc) {
    auto&& [x, y, _] = loc;
    return 3. * y + 1;
});

constexpr auto g2_dy = vs::transform([](auto&& loc) {
    auto&& [x, y, _] = loc;
    return 3. * x + 1;
});

constexpr auto g2_ddx = vs::transform([](auto&& loc) {
    auto&& [_, y, __] = loc;
    return 0;
});

constexpr auto g2_ddy = vs::transform([](auto&& loc) {
    auto&& [x, _, __] = loc;
    return 0;
});

TEST_CASE("E2_2 Domain")
{
    using T = std::vector<real>;

    const auto extents = int3{5, 6, 7};

    auto m = mesh{index_extents{extents},
                  domain_extents{.min = {0.1, 0.2, 0.3}, .max = {1, 2, 2.2}}};

    const auto objectBcs = bcs::Object{};
    const auto loc = m.xyz;

    // initialize fields
    scalar<T> u{loc | f2};

    SECTION("DDFFFD")
    {

        const auto gridBcs = bcs::Grid{bcs::dd, bcs::ff, bcs::fd};
        // set the exact du we expect based on zeros assigned to dirichlet locations
        scalar<T> ex = (loc | f2_ddx) + (loc | f2_ddy) + (loc | f2_ddz);

        // zero boundaries
        ex | m.dirichlet(gridBcs) = 0;

        scalar<T> du{u};
        REQUIRE((integer)rs::size(du | sel::D) == m.size());

        auto lap = laplacian{m, stencils::second::E2, gridBcs, objectBcs};
        du = lap(u);

        REQUIRE_THAT(get<si::D>(ex), Approx(get<si::D>(du)));
    }

    SECTION("DDFFND")
    {

        const auto gridBcs = bcs::Grid{bcs::dd, bcs::ff, bcs::nd};
        // set the exact du we expect based on zeros assigned to dirichlet locations
        scalar<T> ex = (loc | f2_ddx) + (loc | f2_ddy) + (loc | f2_ddz);

        // zero boundaries
        ex | m.dirichlet(gridBcs) = 0;

        // neumann
        scalar<T> nu{loc | f2_dz};

        scalar<T> du{m.ss()};
        REQUIRE((integer)rs::size(du | sel::D) == m.size());

        auto lap = laplacian{m, stencils::second::E2, gridBcs, objectBcs};
        du = lap(u, nu);

        REQUIRE_THAT(get<si::D>(ex), Approx(get<si::D>(du)));
    }
}

TEST_CASE("E2 with Dirichlet Objects")
{
    using T = std::vector<real>;

    sol::state lua;
    lua.script(R"(
        simulation = {
            mesh = {
                index_extents = {25, 26, 27},
                domain_bounds = {
                    min = {0.1, 0.2, 0.3},
                    max = {1, 2, 2.2}
                }
            },
            domain_boundaries = {
                xmin = "dirichlet",
                ymin = "neumann",
                ymax = "neumann",
                zmax = "dirichlet"
            },
            shapes = {
                {
                    type = "sphere",
                    center = {0.45, 1.011, 1.31},
                    radius = 0.25,
                    boundary_condition = "dirichlet"
                }
            },
            scheme = {
                order = 2,
                type = "E2"
            }
        }
    )");
    // mesh
    auto m_opt = mesh::from_lua(lua["simulation"]);
    REQUIRE(!!m_opt);
    const mesh& m = *m_opt;

    // bcs
    auto bc_opt = bcs::from_lua(lua["simulation"], m.extents());
    REQUIRE(!!bc_opt);
    auto&& [gridBcs, objectBcs] = *bc_opt;

    // scheme
    auto scheme_opt = stencil::from_lua(lua["simulation"]);
    REQUIRE(!!scheme_opt);
    stencil st = *scheme_opt;

    const auto loc = m.xyz;

    // initialize fields
    scalar<T> u{loc | f2};
    REQUIRE(rs::size(u | sel::Rx) == m.Rx().size());

    // set the exact du we expect based on zeros assigned to dirichlet locations
    scalar<T> ex{m.ss()};

    ex | m.fluid = (loc | f2_ddx) + (loc | f2_ddy) + (loc | f2_ddz);

    // zero dirichlet boundaries
    ex | m.dirichlet(gridBcs) = 0;

    // neumann conditions
    scalar<T> nu{loc | f2_dy};

    scalar<T> du{m.ss()};

    auto lap = laplacian{m, st, gridBcs, objectBcs};
    du = lap(u, nu);

    REQUIRE_THAT(get<si::D>(ex), Approx(get<si::D>(du)));
}

TEST_CASE("E2 with Floating Objects")
{
    using T = std::vector<real>;

    sol::state lua;
    lua.script(R"(
        simulation = {
            mesh = {
                index_extents = {25, 26, 27},
                domain_bounds = {
                    min = {0.1, 0.2, 0.3},
                    max = {1, 2, 2.2}
                }
            },
            domain_boundaries = {
                xmin = "dirichlet",
                ymin = "neumann",
                ymax = "neumann",
                zmax = "dirichlet"
            },
            shapes = {
                {
                    type = "sphere",
                    center = {0.45, 1.011, 1.31},
                    radius = 0.25,
                    boundary_condition = "floating"
                }
            },
            scheme = {
                order = 2,
                type = "E2"
            }
        }
    )");
    // mesh
    auto m_opt = mesh::from_lua(lua["simulation"]);
    REQUIRE(!!m_opt);
    const mesh& m = *m_opt;

    // bcs
    auto bc_opt = bcs::from_lua(lua["simulation"], m.extents());
    REQUIRE(!!bc_opt);
    auto&& [gridBcs, objectBcs] = *bc_opt;

    // scheme
    auto scheme_opt = stencil::from_lua(lua["simulation"]);
    REQUIRE(!!scheme_opt);
    stencil st = *scheme_opt;

    const auto loc = m.xyz;

    // initialize fields
    scalar<T> u{loc | f2};
    REQUIRE(rs::size(u | sel::Rx) == m.Rx().size());

    // set the exact du we expect based on zeros assigned to dirichlet locations
    scalar<T> ex{m.ss()};

    ex | m.fluid_all(objectBcs) = (loc | f2_ddx) + (loc | f2_ddy) + (loc | f2_ddz);

    // zero dirichlet boundaries
    ex | m.dirichlet(gridBcs, objectBcs) = 0;

    // neumann conditions
    scalar<T> nu{loc | f2_dy};

    scalar<T> du{m.ss()};

    auto lap = laplacian{m, st, gridBcs, objectBcs};
    du = lap(u, nu);

    REQUIRE_THAT(get<si::D>(ex), Approx(get<si::D>(du)));
    REQUIRE_THAT(get<si::Rx>(ex), Approx(get<si::Rx>(du)));
    REQUIRE_THAT(get<si::Ry>(ex), Approx(get<si::Ry>(du)));
    REQUIRE_THAT(get<si::Rz>(ex), Approx(get<si::Rz>(du)));
}

TEST_CASE("2D E2 with Floating Objects")
{
    using T = std::vector<real>;

    sol::state lua;
    lua.script(R"(
        simulation = {
            mesh = {
                index_extents = {25, 26},
                domain_bounds = {
                    min = {0.1, 0.2},
                    max = {1, 2}
                }
            },
            domain_boundaries = {
                xmin = "dirichlet",
                ymin = "neumann",
                ymax = "neumann",
            },
            shapes = {
                {
                    type = "sphere",
                    center = {0.45, 1.011},
                    radius = 0.25,
                    boundary_condition = "floating"
                }
            },
            scheme = {
                order = 2,
                type = "E2"
            }
        }
    )");
    // mesh
    auto m_opt = mesh::from_lua(lua["simulation"]);
    REQUIRE(!!m_opt);
    const mesh& m = *m_opt;

    REQUIRE(m.R(2).size() == 0);

    // bcs
    auto bc_opt = bcs::from_lua(lua["simulation"], m.extents());
    REQUIRE(!!bc_opt);
    auto&& [gridBcs, objectBcs] = *bc_opt;

    // scheme
    auto scheme_opt = stencil::from_lua(lua["simulation"]);
    REQUIRE(!!scheme_opt);
    stencil st = *scheme_opt;

    const auto loc = m.xyz;

    // initialize fields
    scalar<T> u{loc | g2};
    REQUIRE(rs::size(u | sel::Rx) == m.Rx().size());

    // set the exact du we expect based on zeros assigned to dirichlet locations
    scalar<T> ex{m.ss()};

    ex | m.fluid_all(objectBcs) = (loc | g2_ddx) + (loc | g2_ddy);

    // zero dirichlet boundaries
    ex | m.dirichlet(gridBcs, objectBcs) = 0;

    // neumann conditions
    scalar<T> nu{loc | g2_dy};

    scalar<T> du{m.ss()};

    auto lap = laplacian{m, st, gridBcs, objectBcs};
    du = lap(u, nu);

    ex += 1;
    du += 1;

    REQUIRE_THAT(get<si::D>(ex), Approx(get<si::D>(du)));
    REQUIRE_THAT(get<si::Rx>(ex), Approx(get<si::Rx>(du)));
    REQUIRE_THAT(get<si::Ry>(ex), Approx(get<si::Ry>(du)));
}
