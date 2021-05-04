#include "dense.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <range/v3/all.hpp>

#include "random/random.hpp"

using namespace ccs;
using Catch::Matchers::Approx;

TEST_CASE("Identity")
{
    using T = std::vector<real>;

    T imat{1, 0, 0, 0, 0, /* r1 */
           0, 1, 0, 0, 0, /* r2 */
           0, 0, 1, 0, 0, /* r3 */
           0, 0, 0, 1, 0, /* r4 */
           0, 0, 0, 0, 1};
    const auto mat = matrix::dense{5, 5, imat};

    const T rng{-2.0, -1.0, 0.0, 1.0, 2.0};
    T rhs(rng.size());

    mat(rng, rhs);

    REQUIRE_THAT(rng, Approx(rhs));

    mat(rng, rhs, plus_eq);
    const T rng2 = rng | vs::transform([](auto&& x) { return x + x; }) | rs::to<T>();
    REQUIRE_THAT(rng2, Approx(rhs));
}

TEST_CASE("Identity - NonSquare")
{
    using T = std::vector<real>;

    T imat{0, 1, 0, 0, 0, /* r1 */
           0, 0, 1, 0, 0, /* r2 */
           0, 0, 0, 1, 0, /* r3 */
           0, 0, 0, 0, 1};
    const auto A = matrix::dense{4, 5, 1, 0, 1, imat};

    const T x = vs::generate_n([]() { return pick(); }, A.columns()) | rs::to<T>();
    auto b = T(A.columns());

    A(x, b);

    REQUIRE(b[0] == 0.0);

    const auto xx = x | vs::drop(1) | rs::to<T>();
    const auto bb = b | vs::drop(1) | rs::to<T>();
    REQUIRE_THAT(bb, Approx(xx));
}

TEST_CASE("Non Square1")
{
    auto coeffs = std::vector{
        83.95087197745517,  54.60811141907641,   -31.02615838268912, -23.749193978143865,
        -47.73731285639923, 68.77294874944397,   -84.32582368924861, -68.68982276024278,
        -66.25620963886223, -14.145795257191367, -20.66243053847711, 26.12372414082023,
        -84.95259217482109, -26.51110805056817,  63.255420433843824, -62.63678283082555,
        -44.15912219148467, 79.63870038246813,   40.13227588220241,  4.250640715510485,
        27.337743054200416, 64.34395968254239,   -49.2229980208256,  -28.064364905871287,
        45.086404615059905, -74.58586464667385,  45.51639082665275,  10.633368817791506,
        27.62738451878431,  -30.951742698251962, 88.544956462669,    99.34593385752356,
        2.3159193728828313, -75.82935876599558,  10.567250113100783};

    const auto A = matrix::dense{5, 7, coeffs};

    const auto x = std::vector{91.37999616851539,
                               50.64693725996551,
                               -64.60583179621301,
                               -21.187021053699596,
                               61.80093732179881,
                               -51.51860192935814,
                               97.12696476389027};
    auto b = std::vector<real>(A.rows());

    A(x, b);

    REQUIRE_THAT(b,
                 Approx(std::vector{-1738.7987608832982,
                                    -4864.707653579187,
                                    8690.006071322056,
                                    -1676.9985963825454,
                                    -1792.272168705312}));
}

TEST_CASE("Non Square2")
{
    auto coeffs = std::vector{
        -62.96028312265989, -34.222115425099105, -2.9862605513470157, 23.06985790293345,
        -76.73747513448777, 0.9629408474623347,  -40.620449988213124, -55.87954317817193,
        21.40997482658412,  37.708749410667,     -99.44892107309963,  57.76722097490671,
        -32.31615638733223, -16.107987296727003, 3.2555572175240286,  92.33047063004568,
        -73.78219097339218, -99.4087327368714,   -47.41071272784896,  67.99839324859943,
        -56.75967760541846, -30.570447576312347, -63.39035198957018,  -24.724532373486397,
        43.50746274626897,  24.831628943793476,  -67.23007810345939};

    const auto A = matrix::dense{9, 3, coeffs};

    const auto x = std::vector{-71.9024502289571, 70.422079255705, -3.7031798936135942};
    auto b = std::vector<real>(A.rows());

    A(x, b);

    REQUIRE_THAT(b,
                 Approx(std::vector{2128.0647588947263,
                                    -7066.357808643435,
                                    -1093.7287232110582,
                                    -9928.67369062508,
                                    1177.1969541409073,
                                    -11466.553949160041,
                                    8407.725947723382,
                                    -2174.430915359314,
                                    -1130.6331596949249}));
}

TEST_CASE("strided")
{
    const auto coeffs = vs::iota(0, 25);

    for (integer offset = 0; offset < 3; offset++) {
        // setup strided problem
        const auto A = matrix::dense(5, 5, offset, offset, 3, coeffs);
        const auto x = vs::iota(0, 15) | rs::to<std::vector<real>>();
        auto b = std::vector<real>(x.size());
        A(x, b);

        // non-strided problem
        const auto AA = matrix::dense(5, 5, coeffs);
        const auto xx = vs::iota(0, 15) | vs::drop(offset) | vs::stride(3) |
                        rs::to<std::vector<real>>();
        auto bb = std::vector<real>(xx.size());
        AA(xx, bb);

        auto bp = b | vs::drop(offset) | vs::stride(3) | rs::to<std::vector<real>>();
        REQUIRE_THAT(bp, Approx(bb));
    }
}
