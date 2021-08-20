#include "stencil.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <vector>

#include <range/v3/all.hpp>

#include <sol/sol.hpp>
#include <spdlog/spdlog.h>

using Catch::Matchers::Approx;
using namespace ccs;

// interior function for use with E2-poly
constexpr auto gf = [](auto&& x) { return 2 * x * x - 3 * x + 1; };
constexpr auto gt = vs::transform(gf);
constexpr auto g_dx = [](auto&& x) { return 4 * x - 3; };

// boundary function for use with E2-poly
constexpr auto bf = [](auto&& x) { return 2 * x + 1; };
constexpr auto bt = vs::transform(bf);
constexpr auto b_dx = [](auto&& x) { return 2; };

constexpr real ymin = -1.;
constexpr real ymax = 5.;
using T = std::vector<real>;

TEST_CASE("interior")
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(
        simulation = {
            scheme = {
                order = 1,
                type = "E2-poly",
                floating_alpha = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
                dirichlet_alpha = {-0.1, -0.2, -0.3}
            }
        }
    )");

    auto st_opt = stencil::from_lua(lua["simulation"]);
    REQUIRE(!!st_opt);
    const auto& st = *st_opt;

    auto [p, r, t, x] = st.query_max();
    REQUIRE(p == 1);
    REQUIRE(r == 3);
    REQUIRE(t == 4);
    REQUIRE(x == 0);

    const auto mesh = vs::linear_distribute(ymin, ymax, 2 * p + 1) | rs::to<T>();
    const auto h = mesh[1] - mesh[0];
    T c(2 * p + 1);

    REQUIRE(rs::inner_product(st.interior(h, c), mesh | gt, 0.) ==
            Catch::Approx(g_dx(mesh[p])));
}

TEST_CASE("floating")
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(
        simulation = {
            scheme = {
                order = 1,
                type = "E2-poly",
                floating_alpha = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
                dirichlet_alpha = {-0.1, -0.2, -0.3}
            }
        }
    )");

    auto st_opt = stencil::from_lua(lua["simulation"]);
    REQUIRE(!!st_opt);
    const auto& st = *st_opt;

    auto [p, r, t, x] = st.query(bcs::Floating);
    REQUIRE(r == 3);
    REQUIRE(t == 4);

    T c(r * t);
    T ex(x);
    T mesh = vs::linear_distribute(ymin, ymax, t - 1) | rs::to<T>();
    real h = mesh[1] - mesh[0];
    real psi = 0.2;

    {
        T m = vs::concat(vs::single(ymin - psi * h), mesh) | rs::to<T>();
        REQUIRE((int)m.size() == t);

        st.nbs(h, bcs::Floating, psi, false, c, ex);

        for (int i = 0; i < r; i++)
            REQUIRE(rs::inner_product(c | vs::drop(i * t) | vs::take_exactly(t),
                                      m | bt,
                                      0.) == Catch::Approx(b_dx(m[i])));
    }

    {
        T m = vs::concat(mesh, vs::single(ymax + psi * h)) | rs::to<T>();
        REQUIRE((int)m.size() == t);

        st.nbs(h, bcs::Floating, psi, true, c, ex);

        for (int i = 0; i < r; i++)
            REQUIRE(rs::inner_product(c | vs::drop(i * t) | vs::take_exactly(t),
                                      m | bt,
                                      0.) == Catch::Approx(b_dx(m[t - r + i])));
    }
}

TEST_CASE("dirichlet")
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(
        simulation = {
            scheme = {
                order = 1,
                type = "E2-poly",
                floating_alpha = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
                dirichlet_alpha = {-0.1, -0.2, -0.3}
            }
        }
    )");

    auto st_opt = stencil::from_lua(lua["simulation"]);
    REQUIRE(!!st_opt);
    const auto& st = *st_opt;

    auto [p, r, t, x] = st.query(bcs::Dirichlet);
    REQUIRE(r == 2);
    REQUIRE(t == 4);

    T c(r * t);
    T ex(x);
    T mesh = vs::linear_distribute(ymin, ymax, t - 1) | rs::to<T>();
    real h = mesh[1] - mesh[0];
    real psi = 0.0;

    {
        T m = vs::concat(vs::single(ymin - psi * h), mesh) | rs::to<T>();
        REQUIRE((int)m.size() == t);

        st.nbs(h, bcs::Dirichlet, psi, false, c, ex);

        for (int i = 0; i < r; i++)
            REQUIRE(rs::inner_product(c | vs::drop(i * t) | vs::take_exactly(t),
                                      m | bt,
                                      0.) == Catch::Approx(b_dx(m[i + 1])));
    }

    {
        T m = vs::concat(mesh, vs::single(ymax + psi * h)) | rs::to<T>();
        REQUIRE((int)m.size() == t);

        st.nbs(h, bcs::Dirichlet, psi, true, c, ex);

        for (int i = 0; i < r; i++)
            REQUIRE(rs::inner_product(c | vs::drop(i * t) | vs::take_exactly(t),
                                      m | bt,
                                      0.) == Catch::Approx(b_dx(m[t - r - 1 + i])));
    }
}
