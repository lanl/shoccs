#include "Circulant.hpp"

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
#include <range/v3/view/take.hpp>
#include <range/v3/view/transform.hpp>

using namespace ccs;
constexpr auto g = []() { return pick(); };
constexpr auto x2 = [](auto&& x) { return x + x; };

TEST_CASE("Identity")
{
    using T = std::vector<real>;

    T coeffs{1.0};
    randomize();

    {
        const auto A = matrix::Circulant{10, coeffs};
        const auto x = vs::generate_n(g, 10) | rs::to<T>();
        auto b = T(x.size());

        A(x, b);
        REQUIRE(x == b);

        A(x, b, plus_eq);
        const T b2 = x | vs::transform(x2) | rs::to<T>();
        REQUIRE(b2 == b);
    }

    {
        const auto A = matrix::Circulant{10, 1, 2, coeffs};
        const auto x = vs::generate_n(g, 21) | rs::to<T>();
        auto b = T(x.size());

        A(x, b);

        auto q = x | vs::drop(1) | vs::stride(2) | vs::take(A.rows()) | rs::to<T>();
        auto r = b | vs::drop(1) | vs::stride(2) | vs::take(A.rows()) | rs::to<T>();
        REQUIRE(q == r);
    }
}

TEST_CASE("Random")
{
    using Catch::Matchers::Approx;
    using T = std::vector<real>;

    T coeffs{-1.8200787083110566, -4.475169398045676, 0.1576129649934348};

    const T x{-3.1777625401858884,
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

    const auto A = matrix::Circulant{5, 3, 1, coeffs};
    T b(A.rows() + 3);

    A(x, b);
    REQUIRE_THAT(b,
                 Approx(std::vector{0.0,
                                    0.0,
                                    0.0,
                                    44.37616458118171,
                                    -17.087290102627783,
                                    -25.394021164722776,
                                    -27.480391451152123,
                                    23.79690059566026}));
}

TEST_CASE("stride")
{
    using Catch::Matchers::Approx;
    using T = std::vector<real>;

    const T coeffs{-1, 0, 1};

    for (integer offset = 3; offset < 6; offset++) {
        // Set up strided circulant operator
        const auto A = matrix::Circulant(3, offset, 3, coeffs);
        const auto x = vs::iota(0, 15) | rs::to<T>();
        auto b = std::vector<real>(x.size());
        A(x, b);

        // non-strided operator
        const auto AA = matrix::Circulant(3, coeffs);
        const auto xx =
            vs::iota(0, 15) | vs::drop(offset - 3) | vs::stride(3) | rs::to<T>();
        auto bb = T(AA.rows() + 1);
        AA(xx, bb);

        const auto q =
            b | vs::drop(offset) | vs::stride(3) | vs::take(A.rows()) | rs::to<T>();
        const auto r = bb | vs::drop(1) | rs::to<T>();

        REQUIRE_THAT(q, Approx(r));
    }
}
