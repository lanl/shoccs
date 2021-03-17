#include "Block.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <random>
#include <vector>

#include <range/v3/algorithm/equal.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/generate_n.hpp>
#include <range/v3/view/take_exactly.hpp>

TEST_CASE("Identity")
{
    using namespace ccs;
    using namespace ranges::views;

    std::vector<real> left_c{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    std::vector<real> int_c{1.0};
    std::vector<real> right_c{1, 0, 0, 1};

    auto bld = matrix::Block::Builder();
    bld.add_InnerBlock(1,
                       matrix::Dense{4, 4, left_c},
                       matrix::Circulant{4, 10, int_c},
                       matrix::Dense{2, 2, right_c});
    bld.add_InnerBlock(17,
                       matrix::Dense{4, 4, left_c},
                       matrix::Circulant{4, 10, int_c},
                       matrix::Dense{2, 2, right_c});
    bld.add_InnerBlock(34,
                       matrix::Dense{4, 4, left_c},
                       matrix::Circulant{4, 10, int_c},
                       matrix::Dense{2, 2, right_c});

    auto mat = std::move(bld).to_Block(52);

    REQUIRE(mat.rows() == 52);

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis{};
    std::vector<real> rng = generate_n([&gen, &dis]() { return dis(gen); }, mat.rows()) |
                            ranges::to<std::vector<real>>();

    auto res = mat * rng | ranges::to<std::vector<real>>();

    // zero locations
    REQUIRE(res[0] == 0.0);
    REQUIRE(res[33] == 0.0);
    REQUIRE(res[50] == 0.0);
    REQUIRE(res[51] == 0.0);

    REQUIRE(ranges::equal(rng | drop(1) | take(32), res | drop(1) | take(32)));
    REQUIRE(ranges::equal(rng | drop(34) | take(16), res | drop(34) | take(16)));
}

TEST_CASE("Random Boundary")
{
    using namespace ccs;
    using namespace ranges::views;

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
    auto bld = matrix::Block::Builder(3);
    auto msz = 16;

    bld.add_InnerBlock(1,
                       matrix::Dense{4, 5, left_c},
                       matrix::Circulant{3, 10, int_c},
                       matrix::Dense{2, 3, right_c});
    bld.add_InnerBlock(msz + 5,
                       matrix::Dense{4, 5, left_c},
                       matrix::Circulant{3, 10, int_c},
                       matrix::Dense{2, 3, right_c});

    auto mat = std::move(bld).to_Block(2 * msz + 8);

    std::vector<real> rhs_{1.1489955608128035,
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

    auto rhs = concat(single(0.0), rhs_, repeat_n(0.0, 4), rhs_, repeat_n(0.0, 3)) |
               take_exactly(2 * msz + 8);

    auto result = mat * rhs;

    std::vector<real> exact_{109.78978122898833,
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
    auto exact = concat(single(0.0), exact_, repeat_n(0.0, 4), exact_, repeat_n(0.0, 3));

    for (auto&& [comp, ex] : zip(result, exact)) REQUIRE(comp == Catch::Approx(ex));
}
