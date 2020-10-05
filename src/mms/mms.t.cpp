#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "std_matchers.hpp"
#include "manufactured_solutions.hpp"

TEST_CASE("gauss1d")
{
        using namespace ccs;
        using Catch::Matchers::Approx;

        std::vector<real3> center{{1.0}, {2.0}};
        std::vector<real3> variance{{0.5}, {0.3}};
        std::vector<real> amplitude{2.0, 1.2};
        std::vector<real> frequency{0.1, 0.2};

        auto ms = build_ms_gauss(1, center, variance, amplitude, frequency);

        const real3 loc{3.0, 0.0, 0.0};
        const real time = 8.0;

        REQUIRE(ms(time, loc) == Catch::Approx(0.0003319785015967778));

        REQUIRE(ms.ddt(time, loc) == Catch::Approx(-0.000975554445371058));

        const real3 grad{-0.0022343980664847485, 0, 0};
        const auto ms_grad = ms.gradient(time, loc);

        REQUIRE_THAT(ms_grad, Approx(grad));
        REQUIRE(ms.laplacian(time, loc) == Catch::Approx(0.01282798401538059));
}

TEST_CASE("gauss2d")
{
        using namespace ccs;
        using Catch::Matchers::Approx;

        std::vector<real3> center{{1.0, 1.2}, {2.0, -1.0}};
        std::vector<real3> variance{{0.5, 0.8}, {0.3, 0.6}};
        std::vector<real> amplitude{2.0, 1.2};
        std::vector<real> frequency{0.1, 0.2};

        auto ms = build_ms_gauss(2, center, variance, amplitude, frequency);

        const real3 loc{3.0, -0.5, 0.0};
        const real time = 8.0;

        REQUIRE(ms(time, loc) == Catch::Approx(-0.000046838098638663583));

        REQUIRE(ms.ddt(time, loc) == Catch::Approx(-0.0006603967369586969));

        const real3 grad{0.0006725075348924356, 0.0002627963438362986, 0};
        //const auto ms_grad = ms.gradient(time, loc);
        REQUIRE_THAT(ms.gradient(time, loc), Approx(grad));
        REQUIRE(ms.laplacian(time, loc) == Catch::Approx(-0.007471160503672486));
}

TEST_CASE("gauss3d")
{
        using namespace ccs;
        using Catch::Matchers::Approx;

        std::vector<real3> center{{1.0, 1.2, -3.5}, {2.0, -1.0, 0.0}};
        std::vector<real3> variance{{0.5, 0.8, 2.0}, {0.3, 0.6, 0.1}};
        std::vector<real> amplitude{2.0, 1.2};
        std::vector<real> frequency{0.1, 0.2};

        auto ms = build_ms_gauss(3, center, variance, amplitude, frequency);

        const real3 loc{3.0, -0.5, -2.0};
        const real time = 8.0;

        REQUIRE(ms(time, loc) == Catch::Approx(0.00003689973951150938));

        REQUIRE(ms.ddt(time, loc) == Catch::Approx(-3.7993394546164832e-6));

        const real3 grad{
            -0.00029519791609207505, 0.00009801493307744677, -0.000013837402316816019};
        //const auto ms_grad = ms->gradient(time, loc);
        REQUIRE_THAT(ms.gradient(time, loc), Approx(grad));
        REQUIRE(ms.laplacian(time, loc) == Catch::Approx(0.002412644784681726));
}
