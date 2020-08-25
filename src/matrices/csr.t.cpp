#include "csr.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <random>
#include <vector>

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/generate_n.hpp>

TEST_CASE("Identity")
{
    using namespace ccs;

    std::vector<real> w{1, 1, 1, 1, 1};
    std::vector<int> v{0, 1, 2, 3, 4};
    std::vector<int> u{0, 1, 2, 3, 4, 5};

    matrix::csr mat{w, v, u};

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis{};
    std::vector<real> rhs =
        ranges::views::generate_n([&gen, &dis]() { return dis(gen); }, mat.rows()) |
        ranges::to<std::vector<real>>();

    auto res = mat * rhs;

    REQUIRE(res.size() == rhs.size());

    for (int i = 0; i < static_cast<int>(res.size()); i++) REQUIRE(res[i] == rhs[i]);
}

TEST_CASE("Random")
{
    using namespace ccs;

    std::vector<real> w{6.132558989050928,
                        -0.4611523807581932,
                        -2.874686661596037,
                        9.42084557206411,
                        0.2298026797436883,
                        6.066446959605997,
                        -7.70721485928825,
                        -0.9885546582519957,
                        -5.302176517574914};

    std::vector<int> v{1, 6, 0, 4, 6, 7, 8, 9, 0};
    std::vector<int> u{0, 2, 3, 4, 4, 4, 4, 8, 8, 8, 9};

    matrix::csr mat{w, v, u};

    std::vector<real> rhs{-3.612622416000683,
                          -1.1427601879942273,
                          0.8552565615441736,
                          -4.755547850496647,
                          -9.711932690401355,
                          -7.232904766266888,
                          -5.437050079342718,
                          -2.4092054560513105,
                          3.557741982344453,
                          5.127028269794991};

    std::vector<real> exact{-4.500735674823108,
                            10.385157472660014,
                            -91.49461808255228,
                            0.,
                            0.,
                            0.,
                            -48.353395342996556,
                            0.,
                            0.,
                            19.15476174098357};

    auto res = mat * rhs;

    REQUIRE(res.size() == exact.size());

    for (int i = 0; i < static_cast<int>(exact.size()); i++)
        REQUIRE(res[i] == Catch::Approx(exact[i]));
}