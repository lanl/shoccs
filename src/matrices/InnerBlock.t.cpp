#include "InnerBlock.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "random/random.hpp"
#include <vector>

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/drop.hpp>
#include <range/v3/view/generate_n.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/stride.hpp>

TEST_CASE("Identity")
{
    using namespace ccs;
    using Catch::Matchers::Approx;
    using T = std::vector<real>;

    T int_c{1.0};
    T right_c{1, 0, 0, 1};
    randomize();

    SECTION("Square")
    {
        T left_c{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        const auto A = matrix::InnerBlock{16,
                                          0,
                                          0,
                                          1,
                                          matrix::Dense{4, 4, left_c},
                                          matrix::Circulant{10, int_c},
                                          matrix::Dense{2, 2, right_c}};

        REQUIRE(A.rows() == 16);

        const auto x = vs::generate_n([]() { return pick(); }, A.rows()) | rs::to<T>();
        auto b = T(A.rows());

        A(x, b);
        REQUIRE_THAT(x, Approx(b));
    }

    SECTION("Non-Square")
    {
        T left_c{0, 1, 0, 0, 0, /* r */
                 0, 0, 1, 0, 0, /* r */
                 0, 0, 0, 1, 0, /* r */
                 0, 0, 0, 0, 1};

        const auto A = matrix::InnerBlock{16,
                                          1,
                                          0,
                                          1,
                                          matrix::Dense{4, 5, left_c},
                                          matrix::Circulant{9, int_c},
                                          matrix::Dense{2, 2, right_c}};

        REQUIRE(A.rows() == 15);
        REQUIRE(A.columns() == 16);
        REQUIRE(A.col_offset() == 0);
        REQUIRE(A.row_offset() == 1);

        const auto x = vs::generate_n([]() { return pick(); }, A.columns()) | rs::to<T>();
        auto b = T(A.columns());

        A(x, b);

        REQUIRE(b[0] == 0.0);

        const auto xx = x | vs::drop(1) | rs::to<T>();
        const auto bb = b | vs::drop(1) | rs::to<T>();
        REQUIRE_THAT(xx, Approx(bb));
    }
}

TEST_CASE("Random Boundary")
{
    using namespace ccs;
    using Catch::Matchers::Approx;
    using T = std::vector<real>;

    T left_c{
        2.247323503221594,   -5.0275337061235135, -0.9845370624957113, 9.621361506824222,
        -1.4847274599487292, -5.920356924707782,  -2.5821362891416157, 3.4175272453709056,
        8.530925204174977,   6.082313908057927,   -6.47449633704835,   3.2352002101687702,
        4.758308560669512,   1.598859246390333,   0.9341660471574365,  -2.405757184769108,
        -1.0503192473307266, 2.485464032420346,   9.816354187065464,   5.090921023204807};

    T int_c{0.7763383518842382, 2.749405412018632, -0.7966643678385346};

    T right_c{-0.12697902221008128,
              1.4021179149102307,
              0.30625420155821903,
              1.288208276267654,
              1.3345939131355147,
              -1.4713774812532359};

    const auto A = matrix::InnerBlock{16,
                                      2,
                                      2,
                                      1,
                                      matrix::Dense{4, 5, left_c},
                                      matrix::Circulant{10, int_c},
                                      matrix::Dense{2, 3, right_c}};

    const T x{0.0,
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
    T b(x.size());

    A(x, b);

    REQUIRE_THAT(b,
                 Approx(T{0.,
                          0.,
                          109.78978122898833,
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
                          -4.962089239867884}));
}

TEST_CASE("strided")
{
    // Assume we have some field of values on an XxY (15 x 3) grid with the stride in Y ==
    // 1 and we want to compute derivative along the 3 x-lines:
    //
    // 2 5 ... 44
    // 1 4 ... 43
    // 0 3 ... 42

    using namespace ccs;
    using Catch::Matchers::Approx;
    using T = std::vector<real>;

    const T lc = vs::iota(0, 15) | rs::to<T>(); // 3x5 matrix
    const T ic{-2, -1, 0, 1, 2};                // minimum offset of 2 needed
    const T rc = vs::iota(1, 7) | rs::to<T>();  // 2 x 3

    const integer columns = 15;
    const integer stride = 3;

    for (integer offset = 0; offset < 3; offset++) {
        // Set up strided operator
        const auto A = matrix::InnerBlock{columns,
                                          offset,
                                          offset,
                                          stride,
                                          matrix::Dense(3, 5, lc),
                                          matrix::Circulant(10, ic),
                                          matrix::Dense(2, 3, rc)};
        const T x = vs::iota(0, 45) | rs::to<T>();
        T b(x.size());
        A(x, b);

        // non-strided operator
        const auto AA = matrix::InnerBlock(columns,
                                           0,
                                           0,
                                           1,
                                           matrix::Dense(3, 5, lc),
                                           matrix::Circulant(10, ic),
                                           matrix::Dense(2, 3, rc));
        const T xx = vs::iota(0, 45) | vs::drop(offset) | vs::stride(stride) |
                     vs::take(15) | rs::to<T>();
        T bb(xx.size());
        AA(xx, bb);

        const T bp =
            b | vs::drop(offset) | vs::stride(stride) | vs::take(15) | rs::to<T>();

        REQUIRE_THAT(bp, Approx(bb));
    }
}
