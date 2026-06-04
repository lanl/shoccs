#include "circulant.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <algorithm>
#include <ranges>
#include <vector>

#include "fields/lazy_views.hpp"
#include "random/random.hpp"

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

    T coeffs{1.0};
    randomize();

    {
        const auto A = matrix::circulant{10, coeffs};
        T x(10);
        std::generate_n(x.begin(), 10, g);
        auto b = T(x.size());

        A(x, b);
        REQUIRE(x == b);

        A(x, b, plus_eq);
        T b2(x.size());
        std::ranges::transform(x, b2.begin(), x2);
        REQUIRE(b2 == b);
    }

    {
        const auto A = matrix::circulant{10, 1, 2, coeffs};
        T x(21);
        std::generate_n(x.begin(), 21, g);
        auto b = T(x.size());

        A(x, b);

        auto strided_q = ccs::stride(x | std::views::drop(1), 2);
        auto taken_q = strided_q | std::views::take(A.rows());
        T q(taken_q.begin(), taken_q.end());
        auto strided_r = ccs::stride(b | std::views::drop(1), 2);
        auto taken_r = strided_r | std::views::take(A.rows());
        T r(taken_r.begin(), taken_r.end());
        REQUIRE(q == r);
    }
}

TEST_CASE("Random")
{
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

    const auto A = matrix::circulant{5, 3, 1, coeffs};
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
    using T = std::vector<real>;

    const T coeffs{-1, 0, 1};

    for (integer offset = 3; offset < 6; offset++) {
        // Set up strided circulant operator
        const auto A = matrix::circulant(3, offset, 3, coeffs);
        auto iota15 = std::views::iota(0, 15);
        const T x(iota15.begin(), iota15.end());
        auto b = std::vector<real>(x.size());
        A(x, b);

        // non-strided operator
        const auto AA = matrix::circulant(3, coeffs);
        auto strided_xx =
            ccs::stride(std::views::iota(0, 15) | std::views::drop(offset - 3), 3);
        const T xx(strided_xx.begin(), strided_xx.end());
        auto bb = T(AA.rows() + 1);
        AA(xx, bb);

        auto strided_q = ccs::stride(b | std::views::drop(offset), 3);
        auto taken_q = strided_q | std::views::take(A.rows());
        const T q(taken_q.begin(), taken_q.end());
        const T r(bb.begin() + 1, bb.end());

        REQUIRE_THAT(q, Approx(r));
    }
}
