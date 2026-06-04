#include "derivative.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "identity_stencil.hpp"
#include "random/random.hpp"
#include "stencils/stencil.hpp"

#include <algorithm>
#include <ranges>

#include <fmt/core.h>
#include <fmt/ranges.h>

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

constexpr auto f2_ddx = std::views::transform([](auto&& loc) { return 0.0; });
constexpr auto f2_ddy = f2_ddx;
constexpr auto f2_ddz = f2_ddx;

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

void fill_scalar(owned_scalar& s, real val)
{
    std::ranges::fill(s.d_vec, val);
    std::ranges::fill(s.rx_vec, val);
    std::ranges::fill(s.ry_vec, val);
    std::ranges::fill(s.rz_vec, val);
}

void add_offset(owned_scalar& s, real val)
{
    for (auto& v : s.d_vec) v += val;
    for (auto& v : s.rx_vec) v += val;
    for (auto& v : s.ry_vec) v += val;
    for (auto& v : s.rz_vec) v += val;
}

void approx_D(owned_scalar& u, owned_scalar& v)
{
    add_offset(u, 1.0);
    add_offset(v, 1.0);
    REQUIRE_THAT(u.d_vec, Approx(v.d_vec));
    add_offset(u, -1.0);
    add_offset(v, -1.0);
}

void approx_all(owned_scalar& u, owned_scalar& v)
{
    add_offset(u, 1.0);
    add_offset(v, 1.0);
    REQUIRE_THAT(u.d_vec, Approx(v.d_vec));
    REQUIRE_THAT(u.rx_vec, Approx(v.rx_vec));
    REQUIRE_THAT(u.ry_vec, Approx(v.ry_vec));
    REQUIRE_THAT(u.rz_vec, Approx(v.rz_vec));
    add_offset(u, -1.0);
    add_offset(v, -1.0);
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

TEST_CASE("E2_Neumann")
{
    const auto extents = int3{10, 13, 17};

    auto m = mesh{index_extents{extents},
                  domain_extents{.min = {0.1, 0.2, 0.3}, .max = {1, 2, 2.2}}};

    const auto objectBcs = bcs::Object{};

    // initialize fields
    auto u = eval_at_mesh(m, f2);
    auto nu = eval_at_mesh(m, f2_dz);
    auto ex = eval_at_mesh(m, f2_ddz);

    auto du = make_scalar(m);

    const auto gridBcs = bcs::Grid{bcs::dd, bcs::ff, bcs::nn};
    zero_grid_dirichlet(m, gridBcs, ex);

    auto d = derivative(2, m, stencils::second::E2, gridBcs, objectBcs);
    d(u, nu, du);
    // offset for approx
    approx_D(du, ex);

    // since du is zero in this case, add 1 and see if it sticks.
    add_offset(du, 1.0);

    d(u, nu, du, plus_eq);

    // ex *= 2;
    add_offset(ex, 1.0);
    approx_D(du, ex);
}

TEST_CASE("Identity FFFFFF")
{
    const auto extents = int3{5, 7, 6};

    auto m = mesh{index_extents{extents},
                  domain_extents{.min = {-1, -1, 0}, .max = {1, 2, 2.2}}};

    const auto gridBcs = bcs::Grid{bcs::ff, bcs::ff, bcs::ff};
    const auto objectBcs = bcs::Object{};

    // initialize fields
    randomize();
    auto u = make_scalar(m);
    std::ranges::generate(u.d_vec, g);

    auto du = make_scalar(m);
    REQUIRE((integer)du.d_vec.size() == m.size());

    for (int i = 0; i < 3; i++) {
        auto d = derivative{i, m, stencils::identity, gridBcs, objectBcs};
        d(u, du);

        REQUIRE_THAT(u.d_vec, Approx(du.d_vec));
    }
}

TEST_CASE("E2_2 FFFFFF")
{
    const auto extents = int3{5, 7, 6};

    // shift domain bounds away from zero to avoid problems with Catch::Approx
    auto m = mesh{index_extents{extents},
                  domain_extents{.min = {0.1, 0.2, 0.3}, .max = {1, 2, 2.2}}};

    const auto gridBcs = bcs::Grid{bcs::ff, bcs::ff, bcs::ff};
    const auto objectBcs = bcs::Object{};

    // initialize fields
    auto u = eval_at_mesh(m, f2);

    auto du = make_scalar(m);
    REQUIRE((integer)du.d_vec.size() == m.size());

    // exact
    std::array<owned_scalar, 3> dd{
        eval_at_mesh(m, f2_ddx), eval_at_mesh(m, f2_ddy), eval_at_mesh(m, f2_ddz)};

    for (int i = 0; i < 3; i++) {
        auto d = derivative{i, m, stencils::second::E2, gridBcs, objectBcs};
        d(u, du);
        approx_D(dd[i], du);
    }
}

TEST_CASE("Identity Mixed")
{
    const auto extents = int3{5, 7, 6};

    auto m = mesh{index_extents{extents},
                  domain_extents{.min = {-1, -1, 0}, .max = {1, 2, 2.2}}};

    const auto objectBcs = bcs::Object{};

    // initialize fields
    randomize();
    auto u = make_scalar(m);
    std::ranges::generate(u.d_vec, g);

    SECTION("DDFNFD")
    {

        const auto gridBcs = bcs::Grid{bcs::dd, bcs::fn, bcs::fd};
        // set the exact du we expect based on zeros assigned to dirichlet locations
        auto du_exact = copy_scalar(u);
        auto nu = copy_scalar(u);

        // set zeros for dirichlet at xmin/xmax
        zero_grid_dirichlet(m, gridBcs, du_exact);

        auto du = copy_scalar(u);
        REQUIRE((integer)du.d_vec.size() == m.size());

        for (int i = 0; i < m.dims(); i++) {
            auto d = derivative{i, m, stencils::identity, gridBcs, objectBcs};
            fill_scalar(du, 0.0);
            d(u, nu, du);

            REQUIRE_THAT(du_exact.d_vec, Approx(du.d_vec));
        }
    }

    SECTION("NNDDDF")
    {

        const auto gridBcs = bcs::Grid{bcs::nn, bcs::dd, bcs::df};
        // set the exact du we expect based on zeros assigned to dirichlet locations
        auto du_exact = copy_scalar(u);
        auto nu = copy_scalar(u);

        // set zeros for dirichlet at xmin/xmax
        zero_grid_dirichlet(m, gridBcs, du_exact);

        auto du = copy_scalar(u);
        REQUIRE((integer)du.d_vec.size() == m.size());

        for (int i = 0; i < m.dims(); i++) {
            auto d = derivative{i, m, stencils::identity, gridBcs, objectBcs};
            fill_scalar(du, 0.0);
            d(u, nu, du);

            REQUIRE_THAT(du_exact.d_vec, Approx(du.d_vec));
        }
    }
}

TEST_CASE("E2 Mixed")
{
    const auto extents = int3{5, 7, 6};

    auto m = mesh{index_extents{extents},
                  domain_extents{.min = {0.1, 0.2, 0.3}, .max = {1, 2, 2.2}}};

    const auto objectBcs = bcs::Object{};

    // initialize fields
    auto u = eval_at_mesh(m, f2);

    SECTION("DDFFFD")
    {

        const auto gridBcs = bcs::Grid{bcs::dd, bcs::ff, bcs::fd};
        // set the exact du we expect based on zeros assigned to dirichlet locations

        std::array<owned_scalar, 3> dd{
            eval_at_mesh(m, f2_ddx), eval_at_mesh(m, f2_ddy), eval_at_mesh(m, f2_ddz)};
        auto du = make_scalar(m);
        REQUIRE((integer)du.d_vec.size() == m.size());

        for (int i = 0; i < m.dims(); i++) {
            auto d = derivative{i, m, stencils::second::E2, gridBcs, objectBcs};
            fill_scalar(du, 0.0);
            d(u, du);

            auto& ex = dd[i];
            // zero boundaries
            zero_grid_dirichlet(m, gridBcs, ex);

            approx_D(ex, du);
        }
    }

    SECTION("FNDDDF")
    {

        const auto gridBcs = bcs::Grid{bcs::fn, bcs::dd, bcs::df};
        // set the exact du we expect based on zeros assigned to dirichlet locations
        auto nu = eval_at_mesh(m, f2_dx);
        std::array<owned_scalar, 3> dd{
            eval_at_mesh(m, f2_ddx), eval_at_mesh(m, f2_ddy), eval_at_mesh(m, f2_ddz)};

        auto du = copy_scalar(u);
        REQUIRE((integer)du.d_vec.size() == m.size());

        for (int i = 0; i < m.dims(); i++) {
            auto d = derivative{i, m, stencils::second::E2, gridBcs, objectBcs};
            fill_scalar(du, 0.0);
            d(u, nu, du);

            auto& ex = dd[i];
            // zero boundaries
            zero_grid_dirichlet(m, gridBcs, ex);

            approx_D(ex, du);
        }
    }
}

TEST_CASE("Identity with Objects")
{
    const auto extents = int3{16, 19, 18};

    auto m = mesh{index_extents{extents},
                  domain_extents{.min = {-1, -1, 0}, .max = {1, 2, 2.2}},
                  std::vector<shape>{make_sphere(0, real3{0.01, -0.01, 0.99}, 0.25)}};

    const auto gridBcs = bcs::Grid{bcs::ff, bcs::ff, bcs::ff};
    const auto objectBcs = bcs::Object{bcs::Dirichlet};

    // initialize fields
    randomize();
    auto u = eval_at_mesh(m, std::views::transform([](auto&&) { return pick(); }));

    REQUIRE(u.rx_vec.size() == m.Rx().size());

    auto du_x = copy_scalar(u);
    auto du_y = copy_scalar(u);
    auto du_z = copy_scalar(u);
    REQUIRE((integer)du_x.d_vec.size() == m.size());

    auto dx = derivative{0, m, stencils::identity, gridBcs, objectBcs};
    auto dy = derivative{1, m, stencils::identity, gridBcs, objectBcs};
    auto dz = derivative{2, m, stencils::identity, gridBcs, objectBcs};

    fill_scalar(du_x, 0.0);
    dx(u, du_x);
    fill_scalar(du_y, 0.0);
    dy(u, du_y);
    fill_scalar(du_z, 0.0);
    dz(u, du_z);

    REQUIRE_THAT(du_x.d_vec, Approx(du_y.d_vec));
    REQUIRE_THAT(du_x.d_vec, Approx(du_z.d_vec));
}

TEST_CASE("graph matches eager")
{
    const auto extents = int3{5, 7, 6};
    auto m = mesh{index_extents{extents},
                  domain_extents{.min = {-1, -1, 0}, .max = {1, 2, 2.2}}};

    const auto gridBcs = bcs::Grid{bcs::ff, bcs::ff, bcs::ff};
    const auto objectBcs = bcs::Object{};

    randomize();
    auto u = make_scalar(m);
    std::ranges::generate(u.d_vec, g);

    SECTION("Identity non-Neumann eq")
    {
        for (int i = 0; i < 3; i++) {
            auto d = derivative{i, m, stencils::identity, gridBcs, objectBcs};

            auto du_eager = make_scalar(m);
            d(u, du_eager);

            auto du_graph = make_scalar(m);
            d.build_graph(u, du_graph);
            d.submit_graph();

            REQUIRE_THAT(du_graph.d_vec, Approx(du_eager.d_vec));
            REQUIRE_THAT(du_graph.rx_vec, Approx(du_eager.rx_vec));
            REQUIRE_THAT(du_graph.ry_vec, Approx(du_eager.ry_vec));
            REQUIRE_THAT(du_graph.rz_vec, Approx(du_eager.rz_vec));
        }
    }

    SECTION("Identity non-Neumann plus_eq")
    {
        for (int i = 0; i < 3; i++) {
            auto d = derivative{i, m, stencils::identity, gridBcs, objectBcs};

            auto du_eager = make_scalar(m);
            fill_scalar(du_eager, 1.0);
            d(u, du_eager, plus_eq);

            auto du_graph = make_scalar(m);
            fill_scalar(du_graph, 1.0);
            d.build_graph(u, du_graph, plus_eq);
            d.submit_graph();

            REQUIRE_THAT(du_graph.d_vec, Approx(du_eager.d_vec));
        }
    }

    SECTION("graph resubmit produces same result")
    {
        auto d = derivative{0, m, stencils::identity, gridBcs, objectBcs};

        auto du_graph = make_scalar(m);
        d.build_graph(u, du_graph);
        d.submit_graph();
        auto first = du_graph.d_vec;

        fill_scalar(du_graph, 0.0);
        d.submit_graph();

        REQUIRE_THAT(du_graph.d_vec, Approx(first));
    }
}

TEST_CASE("graph matches eager with Neumann")
{
    const auto extents = int3{10, 13, 17};
    auto m = mesh{index_extents{extents},
                  domain_extents{.min = {0.1, 0.2, 0.3}, .max = {1, 2, 2.2}}};

    const auto objectBcs = bcs::Object{};
    const auto gridBcs = bcs::Grid{bcs::dd, bcs::ff, bcs::nn};

    auto u = eval_at_mesh(m, f2);
    auto nu = eval_at_mesh(m, f2_dz);

    auto d = derivative(2, m, stencils::second::E2, gridBcs, objectBcs);

    auto du_eager = make_scalar(m);
    d(u, nu, du_eager);

    auto du_graph = make_scalar(m);
    d.build_graph(u, nu, du_graph);
    d.submit_graph();

    add_offset(du_eager, 1.0);
    add_offset(du_graph, 1.0);
    REQUIRE_THAT(du_graph.d_vec, Approx(du_eager.d_vec));
}

TEST_CASE("E2 with Objects")
{
    const auto extents = int3{25, 26, 27};

    auto m = mesh{index_extents{extents},
                  domain_extents{.min = {0.1, 0.2, 0.3}, .max = {1, 2, 2.2}},
                  std::vector<shape>{make_sphere(0, real3{0.45, 1.011, 1.31}, 0.25)}};

    const auto gridBcs = bcs::Grid{bcs::nn, bcs::dd, bcs::ff};
    const auto objectBcs = bcs::Object{bcs::Floating};

    // initialize fields
    auto u = eval_at_mesh(m, f2);
    REQUIRE(u.rx_vec.size() == m.Rx().size());

    auto nu = eval_at_mesh(m, f2_dx);

    auto du_x = make_scalar(m);
    auto du_y = make_scalar(m);
    auto du_z = make_scalar(m);

    auto ddx = eval_at_mesh(m, f2_ddx);
    auto ddy = eval_at_mesh(m, f2_ddy);
    auto ddz = eval_at_mesh(m, f2_ddz);

    assign_fluid_all(m, objectBcs, du_x, ddx);
    zero_dirichlet(m, gridBcs, objectBcs, du_x);

    assign_fluid_all(m, objectBcs, du_y, ddy);
    zero_dirichlet(m, gridBcs, objectBcs, du_y);

    assign_fluid_all(m, objectBcs, du_z, ddz);
    zero_dirichlet(m, gridBcs, objectBcs, du_z);

    REQUIRE((integer)du_x.d_vec.size() == m.size());

    auto dx = derivative{0, m, stencils::second::E2, gridBcs, objectBcs};
    auto dy = derivative{1, m, stencils::second::E2, gridBcs, objectBcs};
    auto dz = derivative{2, m, stencils::second::E2, gridBcs, objectBcs};

    auto du = copy_scalar(u);

    fill_scalar(du, 0.0);
    dx(u, nu, du);
    approx_all(du, du_x);

    fill_scalar(du, 0.0);
    dy(u, du);
    approx_all(du, du_y);

    fill_scalar(du, 0.0);
    dz(u, du);
    approx_all(du, du_z);
}
