#include "laplacian.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <sol/sol.hpp>
#include <spdlog/spdlog.h>

#include "fields/scalar.hpp"
#include "fields/selection_desc.hpp"
#include "identity_stencil.hpp"
#include "random/random.hpp"
#include "stencils/stencil.hpp"

#include <ranges>

#include <Kokkos_Core.hpp>

// Custom main: Kokkos must be initialized before parallel_for calls.
int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard kokkos(argc, argv);
    return Catch::Session().run(argc, argv);
}

using namespace ccs;
using Catch::Matchers::Approx;

constexpr auto g = []() { return pick(); };

// 2nd order polynomial for use with E2
constexpr auto f2 = std::views::transform([](auto&& loc) {
    auto&& [x, y, z] = loc;
    return x * x * (y + z) + y * y * (x + z) + z * z * (x + y) + 3 * x * y * z + x + y +
           z;
});

constexpr auto f2_dx = std::views::transform([](auto&& loc) {
    auto&& [x, y, z] = loc;
    return 2. * x * (y + z) + y * y + z * z + 3. * y * z + 1;
});

constexpr auto f2_dy = std::views::transform([](auto&& loc) {
    auto&& [x, y, z] = loc;
    return x * x + 2. * y * (x + z) + z * z + 3. * x * z + 1;
});

constexpr auto f2_dz = std::views::transform([](auto&& loc) {
    auto&& [x, y, z] = loc;
    return x * x + y * y + 2. * z * (x + y) + 3. * x * y + 1;
});

constexpr auto f2_ddx = std::views::transform([](auto&& loc) {
    auto&& [_, y, z] = loc;
    return 2. * (y + z);
});

constexpr auto f2_ddy = std::views::transform([](auto&& loc) {
    auto&& [x, _, z] = loc;
    return 2. * (x + z);
});

constexpr auto f2_ddz = std::views::transform([](auto&& loc) {
    auto&& [x, y, _] = loc;
    return 2. * (x + y);
});

// 2D 1st order polynomial for use with E2
constexpr auto g2 = std::views::transform([](auto&& loc) {
    auto&& [x, y, _] = loc;
    return 3 * x * y + x + y + 1;
});

constexpr auto g2_dx = std::views::transform([](auto&& loc) {
    auto&& [x, y, _] = loc;
    return 3. * y + 1;
});

constexpr auto g2_dy = std::views::transform([](auto&& loc) {
    auto&& [x, y, _] = loc;
    return 3. * x + 1;
});

constexpr auto g2_ddx = std::views::transform([](auto&& loc) {
    auto&& [_, y, __] = loc;
    return 0;
});

constexpr auto g2_ddy = std::views::transform([](auto&& loc) {
    auto&& [x, _, __] = loc;
    return 0;
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

void add_offset(owned_scalar& s, real val)
{
    for (auto& v : s.d_vec) v += val;
    for (auto& v : s.rx_vec) v += val;
    for (auto& v : s.ry_vec) v += val;
    for (auto& v : s.rz_vec) v += val;
}

// Element-wise add src into dst.
void add_scalar(owned_scalar& dst, const owned_scalar& src)
{
    for (size_t i = 0; i < dst.d_vec.size(); ++i) dst.d_vec[i] += src.d_vec[i];
    for (size_t i = 0; i < dst.rx_vec.size(); ++i) dst.rx_vec[i] += src.rx_vec[i];
    for (size_t i = 0; i < dst.ry_vec.size(); ++i) dst.ry_vec[i] += src.ry_vec[i];
    for (size_t i = 0; i < dst.rz_vec.size(); ++i) dst.rz_vec[i] += src.rz_vec[i];
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

// Assign src at fluid (D) + non-dirichlet-object (Rx/Ry/Rz) indices.
void assign_fluid_all(const mesh& m,
                      const bcs::Object& o,
                      owned_scalar& dst,
                      const owned_scalar& src)
{
    // D: fluid indices
    auto fd = m.fluid_desc();
    for (int k = 0; k < fd.count(); ++k) {
        int i = fd.element(k);
        dst.d_vec[i] = src.d_vec[i];
    }
    // R: non-dirichlet object indices
    std::vector<real>* dst_R[] = {&dst.rx_vec, &dst.ry_vec, &dst.rz_vec};
    const std::vector<real>* src_R[] = {&src.rx_vec, &src.ry_vec, &src.rz_vec};
    for (int dir = 0; dir < 3; ++dir) {
        auto nd = m.non_dirichlet_object_desc(dir, o);
        for (int k = 0; k < nd.count(); ++k) {
            int i = nd.element(k);
            (*dst_R[dir])[i] = (*src_R[dir])[i];
        }
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

TEST_CASE("E2_2 Domain")
{
    const auto extents = int3{5, 6, 7};

    auto m = mesh{index_extents{extents},
                  domain_extents{.min = {0.1, 0.2, 0.3}, .max = {1, 2, 2.2}}};

    const auto objectBcs = bcs::Object{};

    // initialize fields
    auto u = eval_at_mesh(m, f2);

    SECTION("DDFFFD")
    {

        const auto gridBcs = bcs::Grid{bcs::dd, bcs::ff, bcs::fd};
        // set the exact du we expect based on zeros assigned to dirichlet locations
        auto ex = eval_at_mesh(m, f2_ddx);
        add_scalar(ex, eval_at_mesh(m, f2_ddy));
        add_scalar(ex, eval_at_mesh(m, f2_ddz));

        // zero boundaries
        zero_grid_dirichlet(m, gridBcs, ex);

        auto du = copy_scalar(u);
        REQUIRE((integer)du.d_vec.size() == m.size());

        auto lap = laplacian{m, stencils::second::E2, gridBcs, objectBcs};
        scalar_span du_sp = du;
        du_sp = lap(u);

        REQUIRE_THAT(ex.d_vec, Approx(du.d_vec));
    }

    SECTION("DDFFND")
    {

        const auto gridBcs = bcs::Grid{bcs::dd, bcs::ff, bcs::nd};
        // set the exact du we expect based on zeros assigned to dirichlet locations
        auto ex = eval_at_mesh(m, f2_ddx);
        add_scalar(ex, eval_at_mesh(m, f2_ddy));
        add_scalar(ex, eval_at_mesh(m, f2_ddz));

        // zero boundaries
        zero_grid_dirichlet(m, gridBcs, ex);

        // neumann
        auto nu = eval_at_mesh(m, f2_dz);

        auto du = make_scalar(m);
        REQUIRE((integer)du.d_vec.size() == m.size());

        auto lap = laplacian{m, stencils::second::E2, gridBcs, objectBcs};
        scalar_span du_sp = du;
        du_sp = lap(u, nu);

        REQUIRE_THAT(ex.d_vec, Approx(du.d_vec));
    }
}

TEST_CASE("laplacian graph matches eager")
{
    const auto extents = int3{5, 6, 7};
    auto m = mesh{index_extents{extents},
                  domain_extents{.min = {0.1, 0.2, 0.3}, .max = {1, 2, 2.2}}};
    const auto objectBcs = bcs::Object{};
    auto u = eval_at_mesh(m, f2);

    SECTION("non-Neumann DDFFFD")
    {
        const auto gridBcs = bcs::Grid{bcs::dd, bcs::ff, bcs::fd};

        auto lap = laplacian{m, stencils::second::E2, gridBcs, objectBcs};

        // Eager
        auto du_eager = make_scalar(m);
        scalar_span du_sp_eager = du_eager;
        du_sp_eager = lap(u);

        // Graph
        auto du_graph = make_scalar(m);
        scalar_span du_sp_graph = du_graph;
        lap.build_graph(u, du_sp_graph);
        lap.submit_graph();

        REQUIRE_THAT(du_graph.d_vec, Approx(du_eager.d_vec));
    }

    SECTION("Neumann DDFFND")
    {
        const auto gridBcs = bcs::Grid{bcs::dd, bcs::ff, bcs::nd};
        auto nu = eval_at_mesh(m, f2_dz);

        auto lap = laplacian{m, stencils::second::E2, gridBcs, objectBcs};

        // Eager
        auto du_eager = make_scalar(m);
        scalar_span du_sp_eager = du_eager;
        du_sp_eager = lap(u, nu);

        // Graph
        auto du_graph = make_scalar(m);
        scalar_span du_sp_graph = du_graph;
        lap.build_graph(u, nu, du_sp_graph);
        lap.submit_graph();

        REQUIRE_THAT(du_graph.d_vec, Approx(du_eager.d_vec));
    }

    SECTION("graph resubmit produces same result")
    {
        const auto gridBcs = bcs::Grid{bcs::dd, bcs::ff, bcs::fd};

        auto lap = laplacian{m, stencils::second::E2, gridBcs, objectBcs};

        auto du_graph = make_scalar(m);
        scalar_span du_sp = du_graph;
        lap.build_graph(u, du_sp);
        lap.submit_graph();
        auto first = du_graph.d_vec;

        // Zero and resubmit — same result expected
        std::ranges::fill(du_graph.d_vec, 0.0);
        std::ranges::fill(du_graph.rx_vec, 0.0);
        std::ranges::fill(du_graph.ry_vec, 0.0);
        std::ranges::fill(du_graph.rz_vec, 0.0);
        lap.submit_graph();

        REQUIRE_THAT(du_graph.d_vec, Approx(first));
    }
}

TEST_CASE("E2 with Dirichlet Objects")
{
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

    // initialize fields
    auto u = eval_at_mesh(m, f2);
    REQUIRE(u.rx_vec.size() == m.Rx().size());

    // set the exact du we expect based on zeros assigned to dirichlet locations
    auto sum = eval_at_mesh(m, f2_ddx);
    add_scalar(sum, eval_at_mesh(m, f2_ddy));
    add_scalar(sum, eval_at_mesh(m, f2_ddz));

    auto ex = make_scalar(m);
    assign_fluid(m, ex, sum);

    // zero dirichlet boundaries
    zero_grid_dirichlet(m, gridBcs, ex);

    // neumann conditions
    auto nu = eval_at_mesh(m, f2_dy);

    auto du = make_scalar(m);

    auto lap = laplacian{m, st, gridBcs, objectBcs};
    scalar_span du_sp = du;
    du_sp = lap(u, nu);

    REQUIRE_THAT(ex.d_vec, Approx(du.d_vec));
}

TEST_CASE("E2 with Floating Objects")
{
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

    // initialize fields
    auto u = eval_at_mesh(m, f2);
    REQUIRE(u.rx_vec.size() == m.Rx().size());

    // set the exact du we expect based on zeros assigned to dirichlet locations
    auto sum = eval_at_mesh(m, f2_ddx);
    add_scalar(sum, eval_at_mesh(m, f2_ddy));
    add_scalar(sum, eval_at_mesh(m, f2_ddz));

    auto ex = make_scalar(m);
    assign_fluid_all(m, objectBcs, ex, sum);

    // zero dirichlet boundaries
    zero_dirichlet(m, gridBcs, objectBcs, ex);

    // neumann conditions
    auto nu = eval_at_mesh(m, f2_dy);

    auto du = make_scalar(m);

    auto lap = laplacian{m, st, gridBcs, objectBcs};
    scalar_span du_sp = du;
    du_sp = lap(u, nu);

    REQUIRE_THAT(ex.d_vec, Approx(du.d_vec));
    REQUIRE_THAT(ex.rx_vec, Approx(du.rx_vec));
    REQUIRE_THAT(ex.ry_vec, Approx(du.ry_vec));
    REQUIRE_THAT(ex.rz_vec, Approx(du.rz_vec));
}

TEST_CASE("2D E2 with Floating Objects")
{
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

    // initialize fields
    auto u = eval_at_mesh(m, g2);
    REQUIRE(u.rx_vec.size() == m.Rx().size());

    // set the exact du we expect based on zeros assigned to dirichlet locations
    auto sum = eval_at_mesh(m, g2_ddx);
    add_scalar(sum, eval_at_mesh(m, g2_ddy));

    auto ex = make_scalar(m);
    assign_fluid_all(m, objectBcs, ex, sum);

    // zero dirichlet boundaries
    zero_dirichlet(m, gridBcs, objectBcs, ex);

    // neumann conditions
    auto nu = eval_at_mesh(m, g2_dy);

    auto du = make_scalar(m);

    auto lap = laplacian{m, st, gridBcs, objectBcs};
    scalar_span du_sp = du;
    du_sp = lap(u, nu);

    add_offset(ex, 1.0);
    add_offset(du, 1.0);

    REQUIRE_THAT(ex.d_vec, Approx(du.d_vec));
    REQUIRE_THAT(ex.rx_vec, Approx(du.rx_vec));
    REQUIRE_THAT(ex.ry_vec, Approx(du.ry_vec));
}
