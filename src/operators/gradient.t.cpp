#include "gradient.hpp"

#include "fields/selector.hpp"
#include "stencils/stencil.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <range/v3/all.hpp>

#include <sol/sol.hpp>

using namespace ccs;
using Catch::Matchers::Approx;

const std::vector<real> alpha{
    -1.47956280234494, 0.261900367793859, -0.145072532538541, -0.224665713988644};

// 2nd order polynomial for use with E2
constexpr auto f2 = vs::transform([](auto&& loc) {
    auto&& [x, y, z] = loc;
    return x * (y + z) + y * (x + z) + z * (x + y) + 3 * x * y * z;
});

constexpr auto f2_dx = vs::transform([](auto&& loc) {
    auto&& [x, y, z] = loc;
    return 2. * (y + z) + 3. * y * z;
});

constexpr auto f2_dy = vs::transform([](auto&& loc) {
    auto&& [x, y, z] = loc;
    return 2. * (x + z) + 3. * x * z;
});

constexpr auto f2_dz = vs::transform([](auto&& loc) {
    auto&& [x, y, z] = loc;
    return 2. * (x + y) + 3. * x * y;
});

constexpr auto g = vs::transform([](auto&& loc) {
    auto&& [x, y, z] = loc;
    return x * y + (x + y);
});

constexpr auto gx = vs::transform([](auto&& loc) {
    auto&& [x, y, z] = loc;
    return y + 1;
});

constexpr auto gy = vs::transform([](auto&& loc) {
    auto&& [x, y, z] = loc;
    return x + 1;
});

TEST_CASE("E2_1 Domain")
{
    const auto extents = int3{15, 12, 13};

    auto m = mesh{index_extents{extents},
                  domain_extents{.min = {0.1, 0.2, 0.3}, .max = {1, 2, 2.2}}};

    const auto objectBcs = bcs::Object{};
    const auto loc = m.xyz;
    const auto st = stencils::make_E2_1(alpha);

    // initialize fields
    scalar_real u{loc | f2};

    SECTION("DDFFFD")
    {

        const auto gridBcs = bcs::Grid{bcs::dd, bcs::ff, bcs::fd};
        // set the exact du we expect based on zeros assigned to dirichlet locations
        vector_real ex{loc | f2_dx, loc | f2_dy, loc | f2_dz};

        // zero boundaries
        ex | m.dirichlet(gridBcs) = 0;

        vector_real du{u, u, u};
        REQUIRE((integer)rs::size(du | sel::Dx) == m.size());

        auto grad = gradient{m, st, gridBcs, objectBcs};
        du = grad(u);

        REQUIRE_THAT(get<vi::Dx>(ex), Approx(get<vi::Dx>(du)));
        REQUIRE_THAT(get<vi::Dy>(ex), Approx(get<vi::Dy>(du)));
        REQUIRE_THAT(get<vi::Dz>(ex), Approx(get<vi::Dz>(du)));
    }
}

TEST_CASE("E2 with Objects")
{
    sol::state lua;
    lua.script(R"(
        simulation = {
            mesh = {
                index_extents = {31, 32, 33},
                domain_bounds = {
                    min = {0.1, 0.2, 0.3},
                    max = {1, 2, 2.2}
                }
            },
            domain_boundaries = {
                xmin = "dirichlet",
                zmax = "dirichlet"
            },
            shapes = {
                {
                    type = "sphere",
                    center = {0.45, 1.011, 1.31},
                    radius = 0.141,
                    boundary_condition = "dirichlet"
                }
            },
            scheme = {
                order = 1,
                type = "E2",
                alpha = {-1.47956280234494, 0.261900367793859, -0.145072532538541, -0.224665713988644}
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
    scalar_real u{loc | f2};
    REQUIRE(rs::size(u | sel::Rx) == m.Rx().size());

    // set the exact du we expect based on zeros assigned to dirichlet locations
    vector_real ex{m.vs()};

    ex | m.fluid = tuple{loc | f2_dx, loc | f2_dy, loc | f2_dz};

    // zero dirichlet boundaries
    ex | m.dirichlet(gridBcs) = 0;

    vector_real du{m.vs()};

    auto grad = gradient{m, st, gridBcs, objectBcs};
    du = grad(u);

    REQUIRE_THAT(get<vi::Dx>(ex), Approx(get<vi::Dx>(du)));
    REQUIRE_THAT(get<vi::Dy>(ex), Approx(get<vi::Dy>(du)));
    REQUIRE_THAT(get<vi::Dz>(ex), Approx(get<vi::Dz>(du)));
}

TEST_CASE("2D E2 with Objects - Floating")
{
    sol::state lua;
    lua.script(R"(
        simulation = {
            mesh = {
                index_extents = {21, 15},
                domain_bounds = {
                    min = {0.1, 0.2},
                    max = {2, 2}
                }
            },
            domain_boundaries = {
                xmin = "dirichlet",
                ymin = "dirichlet"
            },
            shapes = {
                {
                    type = "sphere",
                    center = {2, 2},
                    radius = 0.541,
                    boundary_condition = "floating"
                },
                {
                    type = "sphere",
                    center = {0.1, 0.2},
                    radius = 0.4,
                    boundary_condition = "dirichlet"
                }

            },
            scheme = {
                order = 1,
                type = "E2",
                alpha = {-1.47956280234494, 0.261900367793859, -0.145072532538541, -0.224665713988644}
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
    scalar_real u{loc | g};
    REQUIRE(rs::size(u | sel::Rx) == m.Rx().size());

    // set the exact du we expect based on zeros assigned to dirichlet locations
    vector_real ex{m.vs()};

    ex | m.fluid = tuple{loc | gx, loc | gy, loc | gx};
    ex | sel::xR = m.vxyz | gx;
    ex | sel::yR = m.vxyz | gy;
    ex | sel::zR = 0;

    // zero dirichlet boundaries
    ex | m.dirichlet(gridBcs, objectBcs) = 0;

    vector_real du{m.vs()};

    auto grad = gradient{m, st, gridBcs, objectBcs};
    du = grad(u);

    REQUIRE_THAT(get<vi::Dx>(ex), Approx(get<vi::Dx>(du)));
    REQUIRE_THAT(get<vi::Dy>(ex), Approx(get<vi::Dy>(du)));

    REQUIRE_THAT(get<vi::xRx>(ex), Approx(get<vi::xRx>(du)));
    REQUIRE_THAT(get<vi::xRy>(ex), Approx(get<vi::xRy>(du)));

    REQUIRE_THAT(get<vi::yRx>(ex), Approx(get<vi::yRx>(du)));
    REQUIRE_THAT(get<vi::yRy>(ex), Approx(get<vi::yRy>(du)));
}

TEST_CASE("E2 with Objects - Floating")
{
    sol::state lua;
    lua.script(R"(
        simulation = {
            mesh = {
                index_extents = {31, 32, 33},
                domain_bounds = {
                    min = {0.1, 0.2, 0.3},
                    max = {1, 2, 2.2}
                }
            },
            domain_boundaries = {
                xmin = "dirichlet",
                zmax = "dirichlet"
            },
            shapes = {
                {
                    type = "sphere",
                    center = {0.45, 1.011, 1.31},
                    radius = 0.141,
                    boundary_condition = "floating"
                }
            },
            scheme = {
                order = 1,
                type = "E2",
                alpha = {-1.47956280234494, 0.261900367793859, -0.145072532538541, -0.224665713988644}
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
    scalar_real u{loc | f2};
    REQUIRE(rs::size(u | sel::Rx) == m.Rx().size());

    // set the exact du we expect based on zeros assigned to dirichlet locations
    vector_real ex{m.vs()};

    ex | m.fluid = tuple{loc | f2_dx, loc | f2_dy, loc | f2_dz};
    ex | sel::xR = m.vxyz | f2_dx;
    ex | sel::yR = m.vxyz | f2_dy;
    ex | sel::zR = m.vxyz | f2_dz;

    // zero dirichlet boundaries
    ex | m.dirichlet(gridBcs, objectBcs) = 0;

    vector_real du{m.vs()};

    auto grad = gradient{m, st, gridBcs, objectBcs};
    du = grad(u);

    REQUIRE_THAT(get<vi::Dx>(ex), Approx(get<vi::Dx>(du)));
    REQUIRE_THAT(get<vi::Dy>(ex), Approx(get<vi::Dy>(du)));
    REQUIRE_THAT(get<vi::Dz>(ex), Approx(get<vi::Dz>(du)));

    REQUIRE_THAT(get<vi::xRx>(ex), Approx(get<vi::xRx>(du)));
    REQUIRE_THAT(get<vi::xRy>(ex), Approx(get<vi::xRy>(du)));
    REQUIRE_THAT(get<vi::xRz>(ex), Approx(get<vi::xRz>(du)));

    REQUIRE_THAT(get<vi::yRx>(ex), Approx(get<vi::yRx>(du)));
    REQUIRE_THAT(get<vi::yRy>(ex), Approx(get<vi::yRy>(du)));
    REQUIRE_THAT(get<vi::yRz>(ex), Approx(get<vi::yRz>(du)));

    REQUIRE_THAT(get<vi::zRx>(ex), Approx(get<vi::zRx>(du)));
    REQUIRE_THAT(get<vi::zRy>(ex), Approx(get<vi::zRy>(du)));
    REQUIRE_THAT(get<vi::zRz>(ex), Approx(get<vi::zRz>(du)));
}
