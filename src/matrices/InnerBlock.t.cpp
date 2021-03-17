#include "InnerBlock.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <random>
#include <vector>

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/generate_n.hpp>

TEST_CASE("Identity")
{
    using namespace ccs;

    std::vector<real> left_c{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    std::vector<real> int_c{1.0};
    std::vector<real> right_c{1, 0, 0, 1};

    auto mat = matrix::InnerBlock{0,
                                  matrix::Dense{4, 4, left_c},
                                  matrix::Circulant{4, 10, int_c},
                                  matrix::Dense{2, 2, right_c}};

    REQUIRE(mat.rows() == 16);

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis{};
    std::vector<real> rng =
        vs::generate_n([&gen, &dis]() { return dis(gen); }, mat.rows()) |
        rs::to<std::vector<real>>();

    auto res = mat * rng;

    REQUIRE(res.size() == 16u);

    for (int i = 0; i < mat.rows(); i++) REQUIRE(rng[i] == res[i]);
}

TEST_CASE("Random Boundary")
{
    using namespace ccs;

    std::vector<real> left_c{
        2.247323503221594,   -5.0275337061235135, -0.9845370624957113, 9.621361506824222,
        -1.4847274599487292, -5.920356924707782,  -2.5821362891416157, 3.4175272453709056,
        8.530925204174977,   6.082313908057927,   -6.47449633704835,   3.2352002101687702,
        4.758308560669512,   1.598859246390333,   0.9341660471574365,  -2.405757184769108,
        -1.0503192473307266, 2.485464032420346,   9.816354187065464,   5.090921023204807};

    std::vector<real> int_c{0.7763383518842382, 2.749405412018632, -0.7966643678385346};

    std::vector<real> right_c{-0.12697902221008128,
                              1.4021179149102307,
                              0.30625420155821903,
                              1.288208276267654,
                              1.3345939131355147,
                              -1.4713774812532359};

    auto mat = matrix::InnerBlock{2,
                                  matrix::Dense{4, 5, left_c},
                                  matrix::Circulant{3, 10, int_c},
                                  matrix::Dense{2, 3, right_c}};

    std::vector<real> rhs{0.0,
                          0.0,
                          1.1489955608128035,
                          -9.641815125514444,
                          0.5975150739415511,
                          5.321795555035614,
                          -5.467954909319733,
                          1.4687325444642383,
                          8.370366165425736,
                          -2.810816575553474,
                          1.815504174076466,
                          1.1769187743854737,
                          -8.870585410511126,
                          -0.14181686526567816,
                          7.93970270970086,
                          0.14183234352470464,
                          -8.054425087325441,
                          -3.809075460321715};

    auto result = mat * rhs;

    std::vector<real> exact{109.78978122898833,
                            32.2780625869145,
                            -32.38838457188155,
                            33.25158538253831,
                            -12.072197714155793,
                            -6.87521436567669,
                            26.39304084900066,
                            -2.6761855166330424,
                            1.8718030426410628,
                            11.712151684565809,
                            -23.362167910509047,
                            -13.601765954771285,
                            21.606370954127627,
                            12.97052379948305,
                            -12.477808805315766,
                            -4.962089239867884};

    REQUIRE(result.size() == exact.size());

    for (int i = 0; i < static_cast<int>(exact.size()); i++)
        REQUIRE(result[i] == Catch::Approx(exact[i]));
}