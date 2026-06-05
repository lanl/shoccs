#include "inner_block.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "random/random.hpp"
#include <vector>

#include <algorithm>
#include <ranges>

#include "fields/lazy_views.hpp"

#include <Kokkos_Core.hpp>

// Custom main: Kokkos must be initialized before parallel_for calls.
int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard kokkos(argc, argv);
    return Catch::Session().run(argc, argv);
}

using namespace ccs;
using Catch::Matchers::Approx;

constexpr auto g = []() { return pick(); };
constexpr auto x2 = [](auto&& x) { return x + x; };

TEST_CASE("Identity")
{
    using T = std::vector<real>;

    T int_c{1.0};
    T right_c{1, 0, 0, 1};
    randomize();

    SECTION("Square")
    {
        T left_c{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        const auto A = matrix::inner_block{16,
                                           0,
                                           0,
                                           1,
                                           matrix::dense{4, 4, left_c},
                                           matrix::circulant{10, int_c},
                                           matrix::dense{2, 2, right_c}};

        REQUIRE(A.rows() == 16);

        T x(A.rows());
        std::generate_n(x.begin(), A.rows(), g);
        auto b = T(A.rows());

        A(x, b);
        REQUIRE_THAT(x, Approx(b));

        A(x, b, plus_eq);
        T b2(x.size());
        std::ranges::transform(x, b2.begin(), x2);
        REQUIRE_THAT(b2, Approx(b));
    }

    SECTION("Non-Square")
    {
        T left_c{0, 1, 0, 0, 0, /* r */
                 0, 0, 1, 0, 0, /* r */
                 0, 0, 0, 1, 0, /* r */
                 0, 0, 0, 0, 1};

        const auto A = matrix::inner_block{16,
                                           1,
                                           0,
                                           1,
                                           matrix::dense{4, 5, left_c},
                                           matrix::circulant{9, int_c},
                                           matrix::dense{2, 2, right_c}};

        REQUIRE(A.rows() == 15);
        REQUIRE(A.columns() == 16);
        REQUIRE(A.col_offset() == 0);
        REQUIRE(A.row_offset() == 1);

        T x(A.columns());
        std::generate_n(x.begin(), A.columns(), []() { return pick(); });
        auto b = T(A.columns());

        A(x, b);

        REQUIRE(b[0] == 0.0);

        const T xx(x.begin() + 1, x.end());
        const T bb(b.begin() + 1, b.end());
        REQUIRE_THAT(xx, Approx(bb));
    }
}

TEST_CASE("Random Boundary")
{
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

    const auto A = matrix::inner_block{16,
                                       2,
                                       2,
                                       1,
                                       matrix::dense{4, 5, left_c},
                                       matrix::circulant{10, int_c},
                                       matrix::dense{2, 3, right_c}};

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

    using T = std::vector<real>;

    auto iota15 = std::views::iota(0, 15);
    const T lc(iota15.begin(), iota15.end());    // 3x5 matrix
    const T ic{-2, -1, 0, 1, 2};                // minimum offset of 2 needed
    auto iota17 = std::views::iota(1, 7);
    const T rc(iota17.begin(), iota17.end());    // 2 x 3

    const integer columns = 15;
    const integer stride = 3;

    for (integer offset = 0; offset < 3; offset++) {
        // Set up strided operator
        const auto A = matrix::inner_block{columns,
                                           offset,
                                           offset,
                                           stride,
                                           matrix::dense(3, 5, lc),
                                           matrix::circulant(10, ic),
                                           matrix::dense(2, 3, rc)};
        auto iota45 = std::views::iota(0, 45);
        const T x(iota45.begin(), iota45.end());
        T b(x.size());
        A(x, b);

        // non-strided operator
        const auto AA = matrix::inner_block(columns,
                                            0,
                                            0,
                                            1,
                                            matrix::dense(3, 5, lc),
                                            matrix::circulant(10, ic),
                                            matrix::dense(2, 3, rc));
        auto strided_xx = ccs::stride(std::views::iota(0, 45) | std::views::drop(offset), stride);
        auto taken_xx = strided_xx | std::views::take(15);
        const T xx(taken_xx.begin(), taken_xx.end());
        T bb(xx.size());
        AA(xx, bb);

        auto strided_bp = ccs::stride(b | std::views::drop(offset), stride);
        auto taken_bp = strided_bp | std::views::take(15);
        const T bp(taken_bp.begin(), taken_bp.end());

        REQUIRE_THAT(bp, Approx(bb));
    }
}
