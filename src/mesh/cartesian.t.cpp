#include "cartesian.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <range/v3/view/single.hpp>

#include <sol/sol.hpp>
#include <spdlog/spdlog.h>

using namespace ccs;

TEST_CASE("mesh api")
{
    auto m = cartesian{};

    SECTION("3d")
    {
        sol::state lua;
        lua.open_libraries(sol::lib::base, sol::lib::math);
        lua.script(R"(
            simulation = {
                mesh = {
                    index_extents = {11, 31, 51},
                    domain_bounds = {
                        min = {0, -1, -2},
                        max = {1, 2, 3}
                    }
                }
            }
        )");

        auto m_opt = cartesian::from_lua(lua["simulation"]);
        REQUIRE(!!m_opt);
        auto&& [n, domain] = *m_opt;
        auto&& [min, max] = domain;

        m = cartesian{n.extents, min, max};

        REQUIRE(m.dims() == 3);

        REQUIRE(m.on_boundary(0, false, {0, 0, 0}));
        REQUIRE(m.on_boundary(0, false, {0, 10, 11}));
        REQUIRE(!m.on_boundary(0, true, {0, 10, 11}));
        REQUIRE(m.on_boundary(0, true, {10, 10, 11}));

        REQUIRE(m.on_boundary(1, false, {0, 0, 0}));
        REQUIRE(m.on_boundary(1, false, {9, 0, 11}));
        REQUIRE(!m.on_boundary(1, true, {0, 10, 11}));
        REQUIRE(m.on_boundary(1, true, {5, 30, 11}));

        REQUIRE(m.on_boundary(2, false, {0, 0, 0}));
        REQUIRE(m.on_boundary(2, false, {5, 10, 0}));
        REQUIRE(!m.on_boundary(2, true, {5, 10, 0}));
        REQUIRE(m.on_boundary(2, true, {0, 10, 50}));

        for (int i = 0; i < m.dims(); i++) {
            auto line = m.line(i);
            REQUIRE(line.min == min[i]);
            REQUIRE(line.max == max[i]);
            REQUIRE(line.h == Catch::Approx(0.1));
            REQUIRE(line.n == n[i]);

            auto f = m.ucf_ijk2dir(i);
            REQUIRE(f(int3{0, 0, 0}) == 0);
            REQUIRE(f(int3{n[0] - 1, n[1] - 1, n[2] - 1}) == m.size() - 1);

            int3 ijk{};
            ijk[i] = n[i] - 1;
            REQUIRE(f(ijk) == n[i] - 1);

            auto [fast, slow] = index::dirs(i);
            int3 pt{ijk[slow], ijk[fast], ijk[i]};
            auto ff = m.ucf_dir(i);
            REQUIRE(ff(pt) == f(ijk));
        }
    }

    SECTION("2d")
    {
        sol::state lua;
        lua.open_libraries(sol::lib::base, sol::lib::math);
        lua.script(R"(
            simulation = {
                mesh = {
                    index_extents = {11, 31},
                    domain_bounds = {
                        min = {0, -1},
                        max = {1, 2}
                    }
                }
            }
        )");

        auto m_opt = cartesian::from_lua(lua["simulation"]);
        REQUIRE(!!m_opt);
        auto&& [n, domain] = *m_opt;
        auto&& [min, max] = domain;

        m = cartesian{n.extents, min, max};

        REQUIRE(m.dims() == 2);

        for (int i = 0; i < m.dims(); i++) {
            auto line = m.line(i);
            REQUIRE(line.min == min[i]);
            REQUIRE(line.max == max[i]);
            REQUIRE(line.h == Catch::Approx(0.1));
            REQUIRE(line.n == n[i]);

            auto f = m.ucf_ijk2dir(i);
            REQUIRE(f(int3{0, 0, 0}) == 0);
            REQUIRE(f(int3{n[0] - 1, n[1] - 1, 0}) == m.size() - 1);

            int3 ijk{};
            ijk[i] = n[i] - 1;
            REQUIRE(f(ijk) == n[i] - 1);

            auto [fast, slow] = index::dirs(i);
            int3 pt{ijk[slow], ijk[fast], ijk[i]};
            auto ff = m.ucf_dir(i);
            REQUIRE(ff(pt) == f(ijk));
        }
    }

    SECTION("1d")
    {
        sol::state lua;
        lua.open_libraries(sol::lib::base, sol::lib::math);
        lua.script(R"(
            simulation = {
                mesh = {
                    index_extents = {21},
                    domain_bounds = {2}
                }
            }
        )");

        auto m_opt = cartesian::from_lua(lua["simulation"]);
        REQUIRE(!!m_opt);
        auto&& [n, domain] = *m_opt;
        auto&& [min, max] = domain;

        m = cartesian{n.extents, min, max};

        REQUIRE(m.dims() == 1);

        for (int i = 0; i < m.dims(); i++) {
            auto line = m.line(i);
            REQUIRE(line.min == min[0]);
            REQUIRE(line.max == max[0]);
            REQUIRE(line.h == Catch::Approx(0.1));
            REQUIRE(line.n == n[0]);

            auto f = m.ucf_ijk2dir(i);
            REQUIRE(f(int3{0, 0, 0}) == 0);
            REQUIRE(f(int3{n[0] - 1, 0, 0}) == m.size() - 1);

            int3 ijk{};
            ijk[i] = n[i] - 1;
            REQUIRE(f(ijk) == n[i] - 1);

            auto [fast, slow] = index::dirs(i);
            int3 pt{ijk[slow], ijk[fast], ijk[i]};
            auto ff = m.ucf_dir(i);
            REQUIRE(ff(pt) == f(ijk));
        }
    }
}
