#include "stencils.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <vector>

#include <range/v3/view/zip.hpp>

TEST_CASE("dirichlet")
{
    using namespace ccs;

    auto st = stencils::make_E2_2();

    auto [p, r, t, x] = st.query_max();
    REQUIRE(p == 1);
    REQUIRE(r == 2);
    REQUIRE(t == 4);
    REQUIRE(x == 2);

    auto q = st.query(bcs::Dirichlet);
    REQUIRE(q.p == p);
    REQUIRE(q.r == r);
    REQUIRE(q.t == t);
    REQUIRE(q.nextra == 0);

    std::vector<real> left(r * t);
    std::vector<real> right(r * t);
    std::vector<real> extra{};

    st.nbs(0.5, bcs::Dirichlet, 0.0, false, left, extra);

    std::vector<real> exact{0., 0., 0., 0., 4., 0., -8., 4.};

    for (auto&& [comp, ex] : ranges::views::zip(left, exact))
        REQUIRE(comp == Catch::Approx(ex));

    st.nbs(0.5, bcs::Dirichlet, 0.9, true, right, extra);
    exact = std::vector<real>{
        0.5241379310344828, 2.610526315789474, -7.2, 4.0653357531760435, 0., 0., 0., 0.};

    for (auto&& [comp, ex] : ranges::views::zip(right, exact))
        REQUIRE(comp == Catch::Approx(ex));
}

TEST_CASE("floating")
{
    using namespace ccs;

    auto st = stencils::make_E2_2();

    auto [p, r, t, x] = st.query(bcs::Floating);
    REQUIRE(p == 1);
    REQUIRE(r == 2);
    REQUIRE(t == 4);
    REQUIRE(x == 0);

    std::vector<real> c(r * t);
    std::vector<real> extra{};

    st.nbs(0.5, bcs::Floating, 0.0, false, c, extra);

    std::vector<real> exact{0., 4., -8., 4., 4., 0., -8., 4.};

    for (auto&& [comp, ex] : ranges::views::zip(c, exact))
        REQUIRE(comp == Catch::Approx(ex));

    st.nbs(0.5, bcs::Floating, 0.5, true, c, extra);
    exact = std::vector<real>{
        2.4, -2.6666666666666665, -4., 4.266666666666667, 3.25, -5.5, 0.25, 2.};

    for (auto&& [comp, ex] : ranges::views::zip(c, exact))
        REQUIRE(comp == Catch::Approx(ex));
}

TEST_CASE("neumann")
{
    using namespace ccs;

    auto st = stencils::make_E2_2();

    auto [p, r, t, x] = st.query(bcs::Neumann);
    REQUIRE(p == 1);
    REQUIRE(r == 2);
    REQUIRE(t == 4);
    REQUIRE(x == 2);

    std::vector<real> c(r * t);
    std::vector<real> extra(x);

    st.nbs(0.5, bcs::Neumann, 0.0, false, c, extra);

    std::vector<real> exact{-8., 0., 8., 0., 0., -8., 8., 0.};
    std::vector<real> exact_extra{-4., -4.};

    for (auto&& [comp, ex] : ranges::views::zip(c, exact))
        REQUIRE(comp == Catch::Approx(ex));
    for (auto&& [comp, ex] : ranges::views::zip(extra, exact_extra))
        REQUIRE(comp == Catch::Approx(ex));

    st.nbs(0.5, bcs::Neumann, 0.8, true, c, extra);
    exact = std::vector<real>{-0.384,
                              4.928,
                              -7.744,
                              3.2,
                              -0.45714285714285713,
                              2.311111111111111,
                              6.4,
                              -8.253968253968255};
    exact_extra = std::vector<real>{0.8, 4.0};

    for (auto&& [comp, ex] : ranges::views::zip(c, exact))
        REQUIRE(comp == Catch::Approx(ex));
    for (auto&& [comp, ex] : ranges::views::zip(extra, exact_extra))
        REQUIRE(comp == Catch::Approx(ex));
}