#include "stencil.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <vector>

#include <numeric>
#include <ranges>
#include <span>

#include "fields/lazy_views.hpp"

#include <sol/sol.hpp>
#include <spdlog/spdlog.h>

using Catch::Matchers::Approx;
using namespace ccs;

// interior function for use with E2-poly
constexpr auto gf = [](auto&& x) { return 2 * x * x - 3 * x + 1; };
const auto gt = std::views::transform(gf);
constexpr auto g_dx = [](auto&& x) { return 4 * x - 3; };

// boundary function for use with E2-poly
constexpr auto bf = [](auto&& x) { return 2 * x + 1; };
const auto bt = std::views::transform(bf);
constexpr auto b_dx = [](auto&& x) { return 2; };

constexpr real ymin = -1.;
constexpr real ymax = 5.;
using T = std::vector<real>;

TEST_CASE("dirichlet coeffs")
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(
        simulation = {
            scheme = {
                order = 1,
                type = "E2-poly",
                floating_alpha = {},
                dirichlet_alpha = {3/25, 13/100, 7/50}
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
    real h = 1.0;
    real psi = 0.001;

    T exact{-0.4395604395604396,
            -0.4166369491239419,
            0.7128343378083233,
            0.14336305087605816,
            -0.4295704295704296,
            -0.435,
            0.7295704295704296,
            0.135};
    st.nbs(h, bcs::Dirichlet, psi, false, c, ex);
    REQUIRE_THAT(c, Approx(exact));
}

TEST_CASE("interior")
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(
        simulation = {
            scheme = {
                order = 1,
                type = "E2-poly",
                floating_alpha = {},
                dirichlet_alpha = {}
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

    const auto mesh = ccs::linear_distribute(ymin, ymax, 2 * p + 1);
    const auto h = mesh[1] - mesh[0];
    T c(2 * p + 1);

    auto coeffs = st.interior(h, c);
    auto view_gt = mesh | gt;
    REQUIRE(std::inner_product(std::ranges::begin(coeffs), std::ranges::end(coeffs), view_gt.begin(), 0.) ==
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
                dirichlet_alpha = {}
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
    T mesh = ccs::linear_distribute(ymin, ymax, t - 1);
    real h = mesh[1] - mesh[0];
    real psi = 0.2;

    {
        T m = {ymin - psi * h};
        m.insert(m.end(), mesh.begin(), mesh.end());
        REQUIRE((int)m.size() == t);

        st.nbs(h, bcs::Floating, psi, false, c, ex);

        auto view_bt = m | bt;
        for (int i = 0; i < r; i++) {
            auto cs = std::span(c).subspan(i * t, t);
            REQUIRE(std::inner_product(cs.begin(), cs.end(), view_bt.begin(), 0.) == Catch::Approx(b_dx(m[i])));
        }
    }

    {
        T m(mesh.begin(), mesh.end());
        m.push_back(ymax + psi * h);
        REQUIRE((int)m.size() == t);

        st.nbs(h, bcs::Floating, psi, true, c, ex);

        auto view_bt = m | bt;
        for (int i = 0; i < r; i++) {
            auto cs = std::span(c).subspan(i * t, t);
            REQUIRE(std::inner_product(cs.begin(), cs.end(), view_bt.begin(), 0.) == Catch::Approx(b_dx(m[t - r + i])));
        }
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
    T mesh = ccs::linear_distribute(ymin, ymax, t - 1);
    real h = mesh[1] - mesh[0];
    real psi = 0.0;

    {
        T m = {ymin - psi * h};
        m.insert(m.end(), mesh.begin(), mesh.end());
        REQUIRE((int)m.size() == t);

        st.nbs(h, bcs::Dirichlet, psi, false, c, ex);

        auto view_bt = m | bt;
        for (int i = 0; i < r; i++) {
            auto cs = std::span(c).subspan(i * t, t);
            REQUIRE(std::inner_product(cs.begin(), cs.end(), view_bt.begin(), 0.) == Catch::Approx(b_dx(m[i + 1])));
        }
    }

    {
        T m(mesh.begin(), mesh.end());
        m.push_back(ymax + psi * h);
        REQUIRE((int)m.size() == t);

        st.nbs(h, bcs::Dirichlet, psi, true, c, ex);

        auto view_bt = m | bt;
        for (int i = 0; i < r; i++) {
            auto cs = std::span(c).subspan(i * t, t);
            REQUIRE(std::inner_product(cs.begin(), cs.end(), view_bt.begin(), 0.) == Catch::Approx(b_dx(m[t - r - 1 + i])));
        }
    }
}

TEST_CASE("interp interior")
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(
        simulation = {
            scheme = {
                order = 1,
                type = "E2-poly"
            }
        }
    )");

    auto st_opt = stencil::from_lua(lua["simulation"]);
    REQUIRE(!!st_opt);
    const auto& st = *st_opt;

    auto p = st.query_interp().p;

    T c(p);
    T mesh = ccs::linear_distribute(ymin, ymax, p);
    real h = mesh[1] - mesh[0];

    for (auto&& y : ccs::linear_distribute(-0.45, 0.45, 11)) {

        int center = 1 - (y > 0);

        auto&& [v, l, r] = st.interp(2,
                                     int3{1, 2, center},
                                     y,
                                     boundary(int3{1, 2, -10}, std::nullopt),
                                     boundary(int3{1, 2, 10}, std::nullopt),
                                     c);

        REQUIRE(!l.object);
        REQUIRE(!r.object);
        auto view_bt = mesh | bt;
        real yi = std::inner_product(v.begin(), v.end(), view_bt.begin(), 0.);
        REQUIRE(yi == Catch::Approx(bf(mesh[center] + y * h)));
    }
}

TEST_CASE("interp wall")
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(
        f = math.random
        simulation = {
            scheme = {
                order = 1,
                type = "E2-poly",
                floating_alpha = {f(), f(), f(), f(), f(), f()},
                interpolant_alpha = {f(), f(), f(), f()}
            }
        }
    )");

    auto st_opt = stencil::from_lua(lua["simulation"]);
    REQUIRE(!!st_opt);
    const auto& st = *st_opt;

    auto t = st.query_max().t;

    T c(t);
    T mesh = ccs::linear_distribute(ymin, ymax, t - 1);
    const real h = mesh[1] - mesh[0];

    // left 0
    {
        const real psi = 0.8, y = 0.3;
        T m = {ymin - psi * h};
        m.insert(m.end(), mesh.begin(), mesh.end());
        auto&& [v, l, r] = st.interp(0,
                                     int3{8, 4, 5},
                                     y,
                                     boundary{int3{8, 4, 5}, object_boundary{0, 1, psi}},
                                     boundary{int3{50, 4, 5}, std::nullopt},
                                     c);
        REQUIRE(l.object);
        REQUIRE(!r.object);
        auto view_bt = m | bt;
        real yi = std::inner_product(v.begin(), v.end(), view_bt.begin(), 0.);
        REQUIRE(yi == Catch::Approx(bf(ymin - h + y * h)));
    }
    // left 1
    {
        const real psi = 0.8, y = -0.2;
        T m = {ymin - psi * h};
        m.insert(m.end(), mesh.begin(), mesh.end());
        auto&& [v, l, r] = st.interp(0,
                                     int3{9, 4, 5},
                                     y,
                                     boundary{int3{8, 4, 5}, object_boundary{0, 1, psi}},
                                     boundary{int3{50, 4, 5}, std::nullopt},
                                     c);
        REQUIRE(l.object);
        REQUIRE(!r.object);
        auto view_bt = m | bt;
        real yi = std::inner_product(v.begin(), v.end(), view_bt.begin(), 0.);
        REQUIRE(yi == Catch::Approx(bf(ymin + y * h)));
    }

    // right 0
    {
        const real psi = 0.9, y = -0.3;
        T m(mesh.begin(), mesh.end());
        m.push_back(ymax + psi * h);
        auto&& [v, l, r] = st.interp(1,
                                     int3{8, 9, 5},
                                     y,
                                     boundary{int3{8, 0, 5}, std::nullopt},
                                     boundary{int3{8, 9, 5}, object_boundary{0, 0, psi}},
                                     c);
        REQUIRE(!l.object);
        REQUIRE(r.object);
        auto view_bt = m | bt;
        real yi = std::inner_product(v.begin(), v.end(), view_bt.begin(), 0.);
        REQUIRE(yi == Catch::Approx(bf(ymax + h + y * h)));
    }

    // right 1
    {
        const real psi = 0.9, y = 0.3;
        T m(mesh.begin(), mesh.end());
        m.push_back(ymax + psi * h);
        auto&& [v, l, r] = st.interp(1,
                                     int3{8, 8, 5},
                                     y,
                                     boundary{int3{8, 0, 5}, std::nullopt},
                                     boundary{int3{8, 9, 5}, object_boundary{0, 0, psi}},
                                     c);
        REQUIRE(!l.object);
        REQUIRE(r.object);
        auto view_bt = m | bt;
        real yi = std::inner_product(v.begin(), v.end(), view_bt.begin(), 0.);
        REQUIRE(yi == Catch::Approx(bf(ymax + y * h)));
    }
}
