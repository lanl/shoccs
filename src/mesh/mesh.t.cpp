#include "mesh.hpp"
#include "fields/selection_desc.hpp"
#include "random/random.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "real3_operators.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <Kokkos_Core.hpp>
#include <sol/sol.hpp>
#include <spdlog/spdlog.h>

#include <ranges>

// Custom main: Kokkos must be initialized before any test creates Views.
int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard kokkos(argc, argv);
    return Catch::Session().run(argc, argv);
}

using namespace ccs;

constexpr auto g = []() { return pick(); };

// Helper: extract gather_selection indices to a host vector.
std::vector<int> to_host_indices(const gather_selection& gs)
{
    auto h = Kokkos::create_mirror_view(gs.indices_);
    Kokkos::deep_copy(h, gs.indices_);
    std::vector<int> result(gs.count());
    for (int i = 0; i < gs.count(); ++i)
        result[i] = h(i);
    return result;
}

TEST_CASE("lines with no cut-cells")
{
    const auto db = domain_extents{.min = {-1, -1, 0}, .max = {1, 2, 2.2}};
    const auto extents = int3{21, 22, 23};

    auto m = mesh{index_extents{extents}, db};

    REQUIRE(extents[1] * extents[2] == (integer)m.lines(0).size());
    REQUIRE(extents[0] * extents[2] == (integer)m.lines(1).size());
    REQUIRE(extents[0] * extents[1] == (integer)m.lines(2).size());

    {
        auto&& [s, l, r] = m.interp_line(0, int3{9, 10, 11});
        REQUIRE(l.mesh_coordinate == int3{0, 10, 11});
        REQUIRE(r.mesh_coordinate == int3{20, 10, 11});
    }
    {
        auto&& [s, l, r] = m.interp_line(1, int3{9, 10, 11});
        REQUIRE(l.mesh_coordinate == int3{9, 0, 11});
        REQUIRE(r.mesh_coordinate == int3{9, 21, 11});
    }
    {
        auto&& [s, l, r] = m.interp_line(2, int3{9, 10, 11});
        REQUIRE(l.mesh_coordinate == int3{9, 10, 0});
        REQUIRE(r.mesh_coordinate == int3{9, 10, 22});
    }
}

TEST_CASE("lines")
{
    const auto db = domain_extents{.min = {-1, -1, 0}, .max = {1, 2, 2.2}};
    const auto extents = int3{21, 22, 23};

    auto m = mesh{index_extents{extents},
                  db,
                  std::vector<shape>{make_sphere(0, real3{0.01, -0.01, 0.5}, 0.25)}};

    // intersections in x from Mathematica
    SECTION("X")
    {
        constexpr integer n_intersections = 26;
        const auto& line = m.lines(0);

        REQUIRE(extents[1] * extents[2] + n_intersections / 2 == (integer)line.size());

        // check first two lines
        REQUIRE(line[0].stride == extents[1] * extents[2]);
        REQUIRE(line[0].start.mesh_coordinate == int3{0, 0, 0});
        REQUIRE(line[0].end.mesh_coordinate == int3{20, 0, 0});
        REQUIRE(line[1].stride == extents[1] * extents[2]);
        REQUIRE(line[1].start.mesh_coordinate == int3{0, 0, 1});
        REQUIRE(line[1].end.mesh_coordinate == int3{20, 0, 1});

        // check last line
        REQUIRE(line.back().stride == extents[1] * extents[2]);
        REQUIRE(line.back().start.mesh_coordinate == int3{0, 21, 22});
        REQUIRE(line.back().end.mesh_coordinate == int3{20, 21, 22});

        // check first/second intersection point at 10,6,3
        {
            const auto& l = line[6 * extents[2] + 3];
            REQUIRE(l.start.mesh_coordinate == int3{0, 6, 3});
            REQUIRE(!l.start.object);
            REQUIRE(l.end.mesh_coordinate == int3{10, 6, 3});
            REQUIRE(l.end.object);
            REQUIRE(l.end.object->object_coordinate == 0);
            REQUIRE(l.end.object->objectID == 0);
            REQUIRE(l.end.object->psi == Catch::Approx(0.40365385103120377));
        }

        {
            auto&& [s, l, r] = m.interp_line(0, int3{9, 6, 3});
            REQUIRE(l.mesh_coordinate == int3{0, 6, 3});
            REQUIRE(r.mesh_coordinate == int3{10, 6, 3});
            REQUIRE(r.object);
            REQUIRE(r.object->object_coordinate == 0);
            REQUIRE(r.object->objectID == 0);
            REQUIRE(r.object->psi == Catch::Approx(0.40365385103120377));
        }

        {
            const auto& l = line[6 * extents[2] + 3 + 1];
            REQUIRE(l.start.mesh_coordinate == int3{10, 6, 3});
            REQUIRE(l.start.object);
            REQUIRE(l.start.object->object_coordinate == 1);
            REQUIRE(l.start.object->objectID == 0);
            REQUIRE(l.start.object->psi == Catch::Approx(0.2036538510312047));
            REQUIRE(l.end.mesh_coordinate == int3{20, 6, 3});
            REQUIRE(!l.end.object);
        }

        {
            auto&& [s, l, r] = m.interp_line(0, int3{11, 8, 6});
            REQUIRE(l.mesh_coordinate == int3{11, 8, 6});
            REQUIRE(r.mesh_coordinate == int3{extents[0] - 1, 8, 6});
            REQUIRE(l.object);
            REQUIRE(l.object->psi == Catch::Approx(0.1931111964292742));
        }
    }

    SECTION("Y")
    {
        constexpr integer n_intersections = 42;
        const auto& line = m.lines(1);

        REQUIRE(extents[0] * extents[2] + n_intersections / 2 == (integer)line.size());

        // check first two lines
        REQUIRE(line[0].stride == extents[2]);
        REQUIRE(line[0].start.mesh_coordinate == int3{0, 0, 0});
        REQUIRE(line[0].end.mesh_coordinate == int3{0, 21, 0});
        REQUIRE(line[1].stride == extents[2]);
        REQUIRE(line[1].start.mesh_coordinate == int3{0, 0, 1});
        REQUIRE(line[1].end.mesh_coordinate == int3{0, 21, 1});

        // check last line
        REQUIRE(line.back().stride == extents[2]);
        REQUIRE(line.back().start.mesh_coordinate == int3{20, 0, 22});
        REQUIRE(line.back().end.mesh_coordinate == int3{20, 21, 22});

        // check first/second intersection point at 8,7,4
        {
            const auto& l = line[8 * extents[2] + 4];
            REQUIRE(l.start.mesh_coordinate == int3{8, 0, 4});
            REQUIRE(!l.start.object);
            REQUIRE(l.end.mesh_coordinate == int3{8, 7, 4});
            REQUIRE(l.end.object);
            REQUIRE(l.end.object->object_coordinate == 0);
            REQUIRE(l.end.object->objectID == 0);
            REQUIRE(l.end.object->psi == Catch::Approx(0.2884394027061823));
        }

        {
            const auto& l = line[8 * extents[2] + 4 + 1];
            REQUIRE(l.start.mesh_coordinate == int3{8, 7, 4});
            REQUIRE(l.start.object);
            REQUIRE(l.start.object->object_coordinate == 1);
            REQUIRE(l.start.object->objectID == 0);
            REQUIRE(l.start.object->psi == Catch::Approx(0.42843940270618086));
            REQUIRE(l.end.mesh_coordinate == int3{8, 21, 4});
            REQUIRE(!l.end.object);
        }

        {
            auto&& [s, l, r] = m.interp_line(1, int3{8, 9, 4});
            REQUIRE(l.mesh_coordinate == int3{8, 7, 4});
            REQUIRE(r.mesh_coordinate == int3{8, extents[1] - 1, 4});
            REQUIRE(l.object);
            REQUIRE(!r.object);
            REQUIRE(l.object->object_coordinate == 1);
            REQUIRE(l.object->psi == Catch::Approx(0.42843940270618086));
        }
    }
    SECTION("Z")
    {
        constexpr integer n_intersections = 28;
        const auto& line = m.lines(2);

        REQUIRE(extents[0] * extents[1] + n_intersections / 2 == (integer)line.size());

        // check first two lines
        REQUIRE(line[0].stride == 1);
        REQUIRE(line[0].start.mesh_coordinate == int3{0, 0, 0});
        REQUIRE(line[0].end.mesh_coordinate == int3{0, 0, 22});
        REQUIRE(line[1].stride == 1);
        REQUIRE(line[1].start.mesh_coordinate == int3{0, 1, 0});
        REQUIRE(line[1].end.mesh_coordinate == int3{0, 1, 22});

        // check last line
        REQUIRE(line.back().stride == 1);
        REQUIRE(line.back().start.mesh_coordinate == int3{20, 21, 0});
        REQUIRE(line.back().end.mesh_coordinate == int3{20, 21, 22});

        // check last intersection points at 12,8,5
        {
            const auto& l = line[12 * extents[1] + 8 + (n_intersections / 2) - 1];
            REQUIRE(l.start.mesh_coordinate == int3{12, 8, 0});
            REQUIRE(!l.start.object);
            REQUIRE(l.end.mesh_coordinate == int3{12, 8, 5});
            REQUIRE(l.end.object);
            REQUIRE(l.end.object->object_coordinate == n_intersections - 2);
            REQUIRE(l.end.object->objectID == 0);
            REQUIRE(l.end.object->psi == Catch::Approx(0.4491194432954615));
        }

        {
            auto&& [s, l, r] = m.interp_line(2, int3{8, 6, 2});
            REQUIRE(l.mesh_coordinate == int3{8, 6, 0});
            REQUIRE(r.mesh_coordinate == int3{8, 6, 5});
            REQUIRE(!l.object);
            REQUIRE(r.object);
            REQUIRE(r.object->object_coordinate == 0);
            REQUIRE(r.object->psi == Catch::Approx(0.7263250848475999));
        }

        {
            const auto& l = line[12 * extents[1] + 8 + (n_intersections / 2)];
            REQUIRE(l.start.mesh_coordinate == int3{12, 8, 5});
            REQUIRE(l.start.object);
            REQUIRE(l.start.object->object_coordinate == n_intersections - 1);
            REQUIRE(l.start.object->objectID == 0);
            REQUIRE(l.start.object->psi == Catch::Approx(0.4491194432954615));
            REQUIRE(l.end.mesh_coordinate == int3{12, 8, 22});
            REQUIRE(!l.end.object);
        }

        {
            auto&& [s, l, r] = m.interp_line(2, int3{12, 8, 10});
            REQUIRE(l.mesh_coordinate == int3{12, 8, 5});
            REQUIRE(r.mesh_coordinate == int3{12, 8, extents[2] - 1});
            REQUIRE(l.object);
            REQUIRE(!r.object);
            REQUIRE(l.object->object_coordinate == (integer)m.Rz().size() - 1);
            REQUIRE(l.object->psi == Catch::Approx(0.4491194432954615));
        }
    }
}

TEST_CASE("selections")
{
    const auto db = domain_extents{.min = {-1, -1, 0}, .max = {1, 2, 2.2}};
    const auto extents = int3{21, 22, 23};

    auto m = mesh{index_extents{extents}, db};

    randomize();
    std::vector<real> u_d(m.size());
    {
        integer i = 0;
        for (auto&& pt : ccs::cartesian_product(m.x(), m.y(), m.z()))
            u_d[i++] = std::get<0>(pt);
    }
    // Without objects, all cells are fluid — fluid_desc covers everything.
    const auto& fd = m.fluid_desc();
    REQUIRE(fd.count() == m.size());
    auto fluid_idx = to_host_indices(fd);
    for (int i = 0; i < fd.count(); ++i)
        REQUIRE(u_d[fluid_idx[i]] == u_d[i]);
}

TEST_CASE("selections with object")
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(
            simulation = {
                mesh = {
                    index_extents = {21, 22, 23},
                    domain_bounds = {
                        min = {-1, -1,   0},
                        max = { 1,  2, 2.2}
                    }
                },
                shapes = {
                    {
                        type = "sphere",
                        center = {0.01, -0.01, 0.5},
                        radius = 0.25
                    }
                }
            }
        )");
    auto m_opt = mesh::from_lua(lua["simulation"]);
    REQUIRE(!!m_opt);
    const mesh& m = *m_opt;

    std::vector<int> u_d(m.size(), 0);
    std::vector<int> u_rx(m.Rx().size(), 0);
    std::vector<int> u_ry(m.Ry().size(), 0);
    std::vector<int> u_rz(m.Rz().size(), 0);

    std::ranges::fill(u_d, -1);

    // Set fluid cells to 1.
    auto fluid_idx = to_host_indices(m.fluid_desc());
    for (auto idx : fluid_idx)
        u_d[idx] = 1;

    {
        const bcs::Object obj_bcs = {bcs::Floating};
        // Floating => no Dirichlet objects => R components unaffected.
        for (int dir = 0; dir < 3; ++dir) {
            auto desc = m.dirichlet_object_desc(dir, obj_bcs);
            auto idx = to_host_indices(desc);
            auto& r = (dir == 0) ? u_rx : (dir == 1) ? u_ry : u_rz;
            for (auto i : idx)
                r[i] = -1;
        }
        REQUIRE(std::ranges::count(u_rx, -1) == 0);
        REQUIRE(std::ranges::count(u_ry, -1) == 0);
        REQUIRE(std::ranges::count(u_rz, -1) == 0);
    }

    {
        const bcs::Object obj_bcs = {bcs::Dirichlet};
        for (int dir = 0; dir < 3; ++dir) {
            auto desc = m.dirichlet_object_desc(dir, obj_bcs);
            auto idx = to_host_indices(desc);
            auto& r = (dir == 0) ? u_rx : (dir == 1) ? u_ry : u_rz;
            for (auto i : idx)
                r[i] = -1;
        }
        REQUIRE(std::ranges::count(u_rx, -1) == (integer)u_rx.size());
        REQUIRE(std::ranges::count(u_ry, -1) == (integer)u_ry.size());
        REQUIRE(std::ranges::count(u_rz, -1) == (integer)u_rz.size());
    }

    auto nsolid = std::ranges::count(u_d, -1);
    REQUIRE(nsolid > 0);
    auto nfluid = std::ranges::count(u_d, 1);
    REQUIRE(nfluid > 0);

    REQUIRE(nfluid + nsolid == m.size());
    REQUIRE(nfluid == m.fluid_desc().count());

    std::vector<int> v_d(m.size(), 0);
    std::vector<int> v_rx(m.Rx().size(), 0);
    std::vector<int> v_ry(m.Ry().size(), 0);
    std::vector<int> v_rz(m.Rz().size(), 0);

    {
        auto center = real3{0.01, -0.01, 0.5};
        integer i = 0;
        for (auto&& pt : ccs::cartesian_product(m.x(), m.y(), m.z())) {
            real3 loc{std::get<0>(pt), std::get<1>(pt), std::get<2>(pt)};
            v_d[i++] = length(loc - center) > 0.25 ? 1 : -1;
        }
    }

    {
        const bcs::Object obj_bcs = {bcs::Dirichlet};
        // All objects are Dirichlet => reset all R to 0.
        for (int dir = 0; dir < 3; ++dir) {
            auto desc = m.dirichlet_object_desc(dir, obj_bcs);
            auto idx = to_host_indices(desc);
            auto& r = (dir == 0) ? u_rx : (dir == 1) ? u_ry : u_rz;
            for (auto i : idx)
                r[i] = 0;
        }
    }

    REQUIRE(u_d == v_d);
    REQUIRE(u_rx == v_rx);
    REQUIRE(u_ry == v_ry);
    REQUIRE(u_rz == v_rz);

    // test assignment: copy fluid elements from u to w
    std::vector<int> w_d(m.size(), -1);
    std::vector<int> w_rx(m.Rx().size(), 0);
    std::vector<int> w_ry(m.Ry().size(), 0);
    std::vector<int> w_rz(m.Rz().size(), 0);

    for (auto idx : fluid_idx)
        w_d[idx] = u_d[idx];

    REQUIRE(w_d == u_d);
    REQUIRE(w_rx == u_rx);
    REQUIRE(w_ry == u_ry);
    REQUIRE(w_rz == u_rz);
}

TEST_CASE("fluid_desc")
{
    const auto db = domain_extents{.min = {-1, -1, 0}, .max = {1, 2, 2.2}};
    const auto extents = int3{21, 22, 23};

    SECTION("no objects - all cells are fluid")
    {
        auto m = mesh{index_extents{extents}, db};

        // The fluid selector should select all cells for a mesh with no objects.
        const auto& fd = m.fluid_desc();
        REQUIRE(fd.count() == m.size());

        // Verify indices are contiguous [0, size).
        auto h = Kokkos::create_mirror_view(fd.indices_);
        Kokkos::deep_copy(h, fd.indices_);
        for (int i = 0; i < fd.count(); ++i)
            REQUIRE(h(i) == i);
    }

    SECTION("with objects - fluid_desc matches geometric classification")
    {
        sol::state lua;
        lua.open_libraries(sol::lib::base, sol::lib::math);
        lua.script(R"(
            simulation = {
                mesh = {
                    index_extents = {21, 22, 23},
                    domain_bounds = {
                        min = {-1, -1,   0},
                        max = { 1,  2, 2.2}
                    }
                },
                shapes = {
                    {
                        type = "sphere",
                        center = {0.01, -0.01, 0.5},
                        radius = 0.25
                    }
                }
            }
        )");
        auto m_opt = mesh::from_lua(lua["simulation"]);
        REQUIRE(!!m_opt);
        const mesh& m = *m_opt;

        // Compute expected fluid indices from geometry: fluid = outside sphere.
        auto center = real3{0.01, -0.01, 0.5};
        std::vector<int> expected_fluid;
        integer i = 0;
        for (auto&& pt : ccs::cartesian_product(m.x(), m.y(), m.z())) {
            real3 loc{std::get<0>(pt), std::get<1>(pt), std::get<2>(pt)};
            if (length(loc - center) > 0.25)
                expected_fluid.push_back(i);
            ++i;
        }

        const auto& fd = m.fluid_desc();
        REQUIRE(fd.count() == (int)expected_fluid.size());

        auto new_indices = to_host_indices(fd);
        REQUIRE(expected_fluid == new_indices);
    }
}

TEST_CASE("dirichlet_object_desc and non_dirichlet_object_desc")
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(
        simulation = {
            mesh = {
                index_extents = {21, 22, 23},
                domain_bounds = {
                    min = {-1, -1,   0},
                    max = { 1,  2, 2.2}
                }
            },
            shapes = {
                {
                    type = "sphere",
                    center = {0.01, -0.01, 0.5},
                    radius = 0.25
                }
            }
        }
    )");
    auto m_opt = mesh::from_lua(lua["simulation"]);
    REQUIRE(!!m_opt);
    const mesh& m = *m_opt;

    SECTION("all Dirichlet - selects all object points")
    {
        const bcs::Object obj_bcs = {bcs::Dirichlet};
        for (int dir = 0; dir < 3; ++dir) {
            auto gd = m.dirichlet_object_desc(dir, obj_bcs);
            REQUIRE(gd.count() == (int)m.R(dir).size());
            // All indices should be 0..count-1
            auto h = Kokkos::create_mirror_view(gd.indices_);
            Kokkos::deep_copy(h, gd.indices_);
            for (int i = 0; i < gd.count(); ++i)
                REQUIRE(h(i) == i);
        }
    }

    SECTION("all Floating - Dirichlet desc selects nothing")
    {
        const bcs::Object obj_bcs = {bcs::Floating};
        for (int dir = 0; dir < 3; ++dir) {
            auto gd = m.dirichlet_object_desc(dir, obj_bcs);
            REQUIRE(gd.count() == 0);
        }
    }

    SECTION("non_dirichlet is complement of dirichlet")
    {
        const bcs::Object obj_bcs = {bcs::Dirichlet};
        for (int dir = 0; dir < 3; ++dir) {
            auto gd = m.dirichlet_object_desc(dir, obj_bcs);
            auto nd = m.non_dirichlet_object_desc(dir, obj_bcs);
            REQUIRE(gd.count() + nd.count() == (int)m.R(dir).size());
        }
    }
}
