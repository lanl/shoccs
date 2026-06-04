#include "gradient.hpp"

#include "fields/scalar.hpp"
#include "fields/selection_desc.hpp"
#include "stencils/stencil.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <ranges>

#include <sol/sol.hpp>

#include <Kokkos_Core.hpp>

// Custom main: Kokkos must be initialized before parallel_for calls.
int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard kokkos(argc, argv);
    return Catch::Session().run(argc, argv);
}

using namespace ccs;
using Catch::Matchers::Approx;

const std::vector<real> alpha{
    -1.47956280234494, 0.261900367793859, -0.145072532538541, -0.224665713988644};

// 2nd order polynomial for use with E2
constexpr auto f2 = std::views::transform([](auto&& loc) {
    auto&& [x, y, z] = loc;
    return x * (y + z) + y * (x + z) + z * (x + y) + 3 * x * y * z;
});

constexpr auto f2_dx = std::views::transform([](auto&& loc) {
    auto&& [x, y, z] = loc;
    return 2. * (y + z) + 3. * y * z;
});

constexpr auto f2_dy = std::views::transform([](auto&& loc) {
    auto&& [x, y, z] = loc;
    return 2. * (x + z) + 3. * x * z;
});

constexpr auto f2_dz = std::views::transform([](auto&& loc) {
    auto&& [x, y, z] = loc;
    return 2. * (x + y) + 3. * x * y;
});

constexpr auto g = std::views::transform([](auto&& loc) {
    auto&& [x, y, z] = loc;
    return x * y + (x + y);
});

constexpr auto gx = std::views::transform([](auto&& loc) {
    auto&& [x, y, z] = loc;
    return y + 1;
});

constexpr auto gy = std::views::transform([](auto&& loc) {
    auto&& [x, y, z] = loc;
    return x + 1;
});

// Owning scalar: 4 vectors with implicit conversion to scalar_view/scalar_span.
struct owned_scalar {
    std::vector<real> d_vec, rx_vec, ry_vec, rz_vec;

    operator scalar_view() const { return {d_vec, rx_vec, ry_vec, rz_vec}; }
    operator scalar_span() { return {d_vec, rx_vec, ry_vec, rz_vec}; }
};

owned_scalar make_scalar(const mesh& m)
{
    return {std::vector<real>(m.size()),
            std::vector<real>(m.Rx().size()),
            std::vector<real>(m.Ry().size()),
            std::vector<real>(m.Rz().size())};
}

owned_scalar copy_scalar(const owned_scalar& src)
{
    return {src.d_vec, src.rx_vec, src.ry_vec, src.rz_vec};
}

// Evaluate a view adaptor at all mesh locations, producing an owned_scalar.
owned_scalar eval_at_mesh(const mesh& m, auto va)
{
    auto result = make_scalar(m);
    auto pos = std::views::transform(&mesh_object_info::position);
    std::ranges::copy(ccs::cartesian_product(m.x(), m.y(), m.z()) | va,
                      result.d_vec.begin());
    std::ranges::copy(m.Rx() | pos | va, result.rx_vec.begin());
    std::ranges::copy(m.Ry() | pos | va, result.ry_vec.begin());
    std::ranges::copy(m.Rz() | pos | va, result.rz_vec.begin());
    return result;
}

// Zero D at grid dirichlet boundary faces.
void zero_grid_dirichlet(const mesh& m, const bcs::Grid& g, owned_scalar& s)
{
    for_each_grid_bc_desc<bcs::Dirichlet>(g, m.extents(), [&](auto desc) {
        for (int k = 0; k < desc.count(); ++k)
            s.d_vec[desc.element(k)] = 0.0;
    });
}

// Zero D at grid dirichlet + Rx/Ry/Rz at object dirichlet.
void zero_dirichlet(const mesh& m,
                    const bcs::Grid& g,
                    const bcs::Object& o,
                    owned_scalar& s)
{
    zero_grid_dirichlet(m, g, s);
    std::vector<real>* R[] = {&s.rx_vec, &s.ry_vec, &s.rz_vec};
    for (int dir = 0; dir < 3; ++dir) {
        auto gd = m.dirichlet_object_desc(dir, o);
        for (int k = 0; k < gd.count(); ++k)
            (*R[dir])[gd.element(k)] = 0.0;
    }
}

// Assign src at fluid (D only) indices.
void assign_fluid(const mesh& m, owned_scalar& dst, const owned_scalar& src)
{
    auto fd = m.fluid_desc();
    for (int k = 0; k < fd.count(); ++k) {
        int i = fd.element(k);
        dst.d_vec[i] = src.d_vec[i];
    }
}

TEST_CASE("E2_1 Domain")
{
    const auto extents = int3{15, 12, 13};

    auto m = mesh{index_extents{extents},
                  domain_extents{.min = {0.1, 0.2, 0.3}, .max = {1, 2, 2.2}}};

    const auto objectBcs = bcs::Object{};
    const auto st = stencils::make_E2_1(alpha);

    // initialize fields
    auto u = eval_at_mesh(m, f2);

    SECTION("DDFFFD")
    {

        const auto gridBcs = bcs::Grid{bcs::dd, bcs::ff, bcs::fd};
        // set the exact du we expect based on zeros assigned to dirichlet locations
        auto ex_x = eval_at_mesh(m, f2_dx);
        auto ex_y = eval_at_mesh(m, f2_dy);
        auto ex_z = eval_at_mesh(m, f2_dz);

        // zero boundaries
        zero_grid_dirichlet(m, gridBcs, ex_x);
        zero_grid_dirichlet(m, gridBcs, ex_y);
        zero_grid_dirichlet(m, gridBcs, ex_z);

        auto du_x = copy_scalar(u);
        auto du_y = copy_scalar(u);
        auto du_z = copy_scalar(u);
        REQUIRE((integer)du_x.d_vec.size() == m.size());

        auto grad = gradient{m, st, gridBcs, objectBcs};
        grad(u)(du_x, du_y, du_z);

        REQUIRE_THAT(ex_x.d_vec, Approx(du_x.d_vec));
        REQUIRE_THAT(ex_y.d_vec, Approx(du_y.d_vec));
        REQUIRE_THAT(ex_z.d_vec, Approx(du_z.d_vec));
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

    // initialize fields
    auto u = eval_at_mesh(m, f2);
    REQUIRE(u.rx_vec.size() == m.Rx().size());

    // set the exact du we expect based on zeros assigned to dirichlet locations
    // 3 scalar components (x, y, z gradient)
    auto ex_x = make_scalar(m);
    auto ex_y = make_scalar(m);
    auto ex_z = make_scalar(m);

    // assign at fluid D indices
    {
        auto src_x = eval_at_mesh(m, f2_dx);
        auto src_y = eval_at_mesh(m, f2_dy);
        auto src_z = eval_at_mesh(m, f2_dz);
        assign_fluid(m, ex_x, src_x);
        assign_fluid(m, ex_y, src_y);
        assign_fluid(m, ex_z, src_z);
    }

    // zero dirichlet boundaries
    zero_grid_dirichlet(m, gridBcs, ex_x);
    zero_grid_dirichlet(m, gridBcs, ex_y);
    zero_grid_dirichlet(m, gridBcs, ex_z);

    auto du_x = make_scalar(m);
    auto du_y = make_scalar(m);
    auto du_z = make_scalar(m);

    auto grad = gradient{m, st, gridBcs, objectBcs};
    grad(u)(du_x, du_y, du_z);

    REQUIRE_THAT(ex_x.d_vec, Approx(du_x.d_vec));
    REQUIRE_THAT(ex_y.d_vec, Approx(du_y.d_vec));
    REQUIRE_THAT(ex_z.d_vec, Approx(du_z.d_vec));
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

    // initialize fields
    auto u = eval_at_mesh(m, g);
    REQUIRE(u.rx_vec.size() == m.Rx().size());

    // set the exact du we expect based on zeros assigned to dirichlet locations
    // 3 scalar components (x, y, z gradient)
    auto ex_x = make_scalar(m);
    auto ex_y = make_scalar(m);
    auto ex_z = make_scalar(m);

    // assign at fluid D indices
    {
        auto src_x = eval_at_mesh(m, gx);
        auto src_y = eval_at_mesh(m, gy);
        auto src_z = eval_at_mesh(m, gx);  // 2D test — z-component not asserted
        assign_fluid(m, ex_x, src_x);
        assign_fluid(m, ex_y, src_y);
        assign_fluid(m, ex_z, src_z);
    }
    // R components: fill from object positions
    {
        auto pos = std::views::transform(&mesh_object_info::position);
        std::ranges::copy(m.Rx() | pos | gx, ex_x.rx_vec.begin());
        std::ranges::copy(m.Ry() | pos | gx, ex_x.ry_vec.begin());
        std::ranges::copy(m.Rz() | pos | gx, ex_x.rz_vec.begin());
        std::ranges::copy(m.Rx() | pos | gy, ex_y.rx_vec.begin());
        std::ranges::copy(m.Ry() | pos | gy, ex_y.ry_vec.begin());
        std::ranges::copy(m.Rz() | pos | gy, ex_y.rz_vec.begin());
    }
    // z R components are already zero from make_scalar

    // zero dirichlet boundaries
    zero_dirichlet(m, gridBcs, objectBcs, ex_x);
    zero_dirichlet(m, gridBcs, objectBcs, ex_y);
    zero_dirichlet(m, gridBcs, objectBcs, ex_z);

    auto du_x = make_scalar(m);
    auto du_y = make_scalar(m);
    auto du_z = make_scalar(m);

    auto grad = gradient{m, st, gridBcs, objectBcs};
    grad(u)(du_x, du_y, du_z);

    REQUIRE_THAT(ex_x.d_vec, Approx(du_x.d_vec));
    REQUIRE_THAT(ex_y.d_vec, Approx(du_y.d_vec));

    REQUIRE_THAT(ex_x.rx_vec, Approx(du_x.rx_vec));
    REQUIRE_THAT(ex_x.ry_vec, Approx(du_x.ry_vec));

    REQUIRE_THAT(ex_y.rx_vec, Approx(du_y.rx_vec));
    REQUIRE_THAT(ex_y.ry_vec, Approx(du_y.ry_vec));
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

    // initialize fields
    auto u = eval_at_mesh(m, f2);
    REQUIRE(u.rx_vec.size() == m.Rx().size());

    // set the exact du we expect based on zeros assigned to dirichlet locations
    // 3 scalar components (x, y, z gradient)
    auto ex_x = make_scalar(m);
    auto ex_y = make_scalar(m);
    auto ex_z = make_scalar(m);

    // assign at fluid D indices
    {
        auto src_x = eval_at_mesh(m, f2_dx);
        auto src_y = eval_at_mesh(m, f2_dy);
        auto src_z = eval_at_mesh(m, f2_dz);
        assign_fluid(m, ex_x, src_x);
        assign_fluid(m, ex_y, src_y);
        assign_fluid(m, ex_z, src_z);
    }
    // R components: fill from object positions
    {
        auto pos = std::views::transform(&mesh_object_info::position);
        std::ranges::copy(m.Rx() | pos | f2_dx, ex_x.rx_vec.begin());
        std::ranges::copy(m.Ry() | pos | f2_dx, ex_x.ry_vec.begin());
        std::ranges::copy(m.Rz() | pos | f2_dx, ex_x.rz_vec.begin());
        std::ranges::copy(m.Rx() | pos | f2_dy, ex_y.rx_vec.begin());
        std::ranges::copy(m.Ry() | pos | f2_dy, ex_y.ry_vec.begin());
        std::ranges::copy(m.Rz() | pos | f2_dy, ex_y.rz_vec.begin());
        std::ranges::copy(m.Rx() | pos | f2_dz, ex_z.rx_vec.begin());
        std::ranges::copy(m.Ry() | pos | f2_dz, ex_z.ry_vec.begin());
        std::ranges::copy(m.Rz() | pos | f2_dz, ex_z.rz_vec.begin());
    }

    // zero dirichlet boundaries
    zero_dirichlet(m, gridBcs, objectBcs, ex_x);
    zero_dirichlet(m, gridBcs, objectBcs, ex_y);
    zero_dirichlet(m, gridBcs, objectBcs, ex_z);

    auto du_x = make_scalar(m);
    auto du_y = make_scalar(m);
    auto du_z = make_scalar(m);

    auto grad = gradient{m, st, gridBcs, objectBcs};
    grad(u)(du_x, du_y, du_z);

    REQUIRE_THAT(ex_x.d_vec, Approx(du_x.d_vec));
    REQUIRE_THAT(ex_y.d_vec, Approx(du_y.d_vec));
    REQUIRE_THAT(ex_z.d_vec, Approx(du_z.d_vec));

    REQUIRE_THAT(ex_x.rx_vec, Approx(du_x.rx_vec));
    REQUIRE_THAT(ex_x.ry_vec, Approx(du_x.ry_vec));
    REQUIRE_THAT(ex_x.rz_vec, Approx(du_x.rz_vec));

    REQUIRE_THAT(ex_y.rx_vec, Approx(du_y.rx_vec));
    REQUIRE_THAT(ex_y.ry_vec, Approx(du_y.ry_vec));
    REQUIRE_THAT(ex_y.rz_vec, Approx(du_y.rz_vec));

    REQUIRE_THAT(ex_z.rx_vec, Approx(du_z.rx_vec));
    REQUIRE_THAT(ex_z.ry_vec, Approx(du_z.ry_vec));
    REQUIRE_THAT(ex_z.rz_vec, Approx(du_z.rz_vec));
}
