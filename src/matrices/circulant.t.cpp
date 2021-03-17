#include "Circulant.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <random>
#include <vector>

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/generate_n.hpp>

TEST_CASE("Identity")
{
    using namespace ccs;

    std::vector<real> coeffs{1.0};

    auto mat = matrix::Circulant{0, 10, coeffs};

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis{};
    std::vector<real> rng =
        ranges::views::generate_n([&gen, &dis]() { return dis(gen); }, 10) |
        ranges::to<std::vector<real>>();

    auto res = mat * rng;

    REQUIRE(res.size() == 10u);

    for (int i = 0; i < 10; i++) REQUIRE(rng[i] == res[i]);
}

TEST_CASE("Random")
{
    using namespace ccs;

    std::vector<real> coeffs{-1.8200787083110566, -4.475169398045676, 0.1576129649934348};

    std::vector<real> rhs{-3.1777625401858884,
                          -1.1637081452907765,
                          -8.784390495345235,
                          -6.1173200419288705,
                          6.419770261059497,
                          3.22476867262354,
                          4.579902923667724,
                          -7.075669447054981,
                          2.9684424313172144,
                          -5.8732966768985975,
                          3.6343912595600116};

    auto mat = matrix::Circulant{2, 5, coeffs};

    auto rng = mat * rhs;

    std::vector<real> exact{44.37616458118171,
                            -17.087290102627783,
                            -25.394021164722776,
                            -27.480391451152123,
                            23.79690059566026};

    REQUIRE(rng.size() == exact.size());

    for (int i = 0; i < 5; i++) REQUIRE(rng[i] == Catch::Approx(exact[i]));
}