#include "stencil.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

/// tension_E4u_1 — uniform-mesh E4-order boundary closure derived from a
/// tension-spline RBF interpolant. Runtime-parameterized by `sigma`, the
/// tension-kernel shape parameter. The constructor solves the small RBF linear
/// system once and caches the resulting 5x7 coefficient block in
/// `cached_coeffs`; `nbs_floating` reads from the cache.

namespace ccs::stencils
{
namespace
{

// Tension spline kernel φ(r;σ) = σ|r| - 1 + exp(-σ|r|).
// Taylor form for small z = σ|r| (matches phs._tension_kernel_eval).
real tension_phi(real r, real sigma)
{
    const real ar = std::abs(r);
    if (ar == 0.0) return 0.0;
    const real z = sigma * ar;
    if (z < 2.0) {
        real s = -1.0 / 181440.0;
        s = 1.0 / 20160.0 + z * s;
        s = -1.0 / 2520.0 + z * s;
        s = 1.0 / 360.0 + z * s;
        s = -1.0 / 60.0 + z * s;
        s = 1.0 / 12.0 + z * s;
        s = -1.0 / 3.0 + z * s;
        s = 1.0 + z * s;
        return 0.5 * z * z * s;
    }
    return z - 1.0 + std::exp(-z);
}

// D^1 φ(r;σ) = σ·sign(r)·(1 - exp(-σ|r|)).
real tension_dphi(real r, real sigma)
{
    if (r == 0.0) return 0.0;
    const real ar = std::abs(r);
    const real sgn = r > 0 ? 1.0 : -1.0;
    const real z = sigma * ar;
    if (z < 2.0) {
        real s = -1.0 / 40320.0;
        s = 1.0 / 5040.0 + z * s;
        s = -1.0 / 720.0 + z * s;
        s = 1.0 / 120.0 + z * s;
        s = -1.0 / 24.0 + z * s;
        s = 1.0 / 6.0 + z * s;
        s = -1.0 / 2.0 + z * s;
        s = 1.0 + z * s;
        return sgn * sigma * z * s;
    }
    return sgn * sigma * (1.0 - std::exp(-z));
}

// Solve A x = b in place via Gaussian elimination with partial pivoting.
// `A` is N×N, `B` is N×NRHS, both stored row-major. On exit `B` holds the
// solution vectors as columns.
template <std::size_t N, std::size_t NRHS>
void gauss_solve(std::array<real, N * N>& A, std::array<real, N * NRHS>& B)
{
    for (std::size_t k = 0; k < N; ++k) {
        // Partial pivot.
        std::size_t piv = k;
        real piv_abs = std::abs(A[k * N + k]);
        for (std::size_t i = k + 1; i < N; ++i) {
            const real v = std::abs(A[i * N + k]);
            if (v > piv_abs) {
                piv_abs = v;
                piv = i;
            }
        }
        if (piv != k) {
            for (std::size_t j = k; j < N; ++j)
                std::swap(A[k * N + j], A[piv * N + j]);
            for (std::size_t j = 0; j < NRHS; ++j)
                std::swap(B[k * NRHS + j], B[piv * NRHS + j]);
        }
        const real akk = A[k * N + k];
        for (std::size_t i = k + 1; i < N; ++i) {
            const real f = A[i * N + k] / akk;
            for (std::size_t j = k; j < N; ++j)
                A[i * N + j] -= f * A[k * N + j];
            for (std::size_t j = 0; j < NRHS; ++j)
                B[i * NRHS + j] -= f * B[k * NRHS + j];
        }
    }
    // Back-substitution.
    for (std::size_t ki = N; ki-- > 0;) {
        const real akk = A[ki * N + ki];
        for (std::size_t j = 0; j < NRHS; ++j) {
            real s = B[ki * NRHS + j];
            for (std::size_t m = ki + 1; m < N; ++m)
                s -= A[ki * N + m] * B[m * NRHS + j];
            B[ki * NRHS + j] = s / akk;
        }
    }
}

// Compute the 5x7 boundary-block coefficients at h=1 for the tension-spline
// closure with the given sigma. Matches phs._rbf_weights_numeric with
// (p=2, q=3, kernel='tension', nu=1, nextra=0): t=6 collocation points at
// x = 0..5 plus a q+1=4 polynomial augmentation. Rows 0..3 are the boundary
// rows obtained from the 10x10 augmented solve. Row 4 is the classical
// E4 centered first-derivative stencil at x=4 on a unit grid.
void solve_tension_coefficients(real sigma, std::array<real, 5 * 7>& out)
{
    constexpr std::size_t T = 6;   // number of collocation points
    constexpr std::size_t Q1 = 4;  // polynomial degree q=3 → q+1 columns
    constexpr std::size_t N = T + Q1;  // augmented system size (10)
    constexpr std::size_t NRHS = 4;    // one RHS per boundary row i = 0..3

    // Collocation points x_j = 0..5 on the unit grid.
    std::array<real, T> pts{0.0, 1.0, 2.0, 3.0, 4.0, 5.0};

    // Build LHS: [Φ P^T; P 0] where Φ_{jk}=φ(x_j-x_k), P_{mj}=x_j^m.
    std::array<real, N * N> A{};
    for (std::size_t j = 0; j < T; ++j) {
        for (std::size_t k = 0; k < T; ++k) {
            A[j * N + k] = tension_phi(pts[j] - pts[k], sigma);
        }
    }
    for (std::size_t m = 0; m < Q1; ++m) {
        for (std::size_t j = 0; j < T; ++j) {
            const real p = std::pow(pts[j], static_cast<int>(m));
            A[j * N + (T + m)] = p;  // P^T upper-right
            A[(T + m) * N + j] = p;  // P lower-left
        }
    }
    // Lower-right Q1×Q1 block is zero (already default-initialized).

    // Build RHS: 4 columns, one per boundary row i in 0..3.
    std::array<real, N * NRHS> B{};
    for (std::size_t i = 0; i < NRHS; ++i) {
        const real xe = static_cast<real>(i);
        for (std::size_t j = 0; j < T; ++j) {
            B[j * NRHS + i] = tension_dphi(xe - pts[j], sigma);
        }
        // D^1 x^m at x=xe: m * xe^(m-1) for m>=1; 0 for m=0.
        for (std::size_t m = 0; m < Q1; ++m) {
            if (m == 0) {
                B[(T + m) * NRHS + i] = 0.0;
            } else {
                B[(T + m) * NRHS + i] =
                    static_cast<real>(m) * std::pow(xe, static_cast<int>(m) - 1);
            }
        }
    }

    gauss_solve<N, NRHS>(A, B);

    // Extract first T entries of each column as the boundary weights; lay
    // out in a 5x7 row-major buffer with col 6 = 0 for rows 0..3.
    std::fill(out.begin(), out.end(), 0.0);
    for (std::size_t i = 0; i < NRHS; ++i) {
        for (std::size_t j = 0; j < T; ++j) {
            out[i * 7 + j] = B[j * NRHS + i];
        }
    }
    // Row 4: classical E4 centered stencil at x=4 on the 7-point support
    // {0,1,2,3,4,5,6}. Coefficients at h=1.
    out[4 * 7 + 0] = 0.0;
    out[4 * 7 + 1] = 0.0;
    out[4 * 7 + 2] = 1.0 / 12.0;
    out[4 * 7 + 3] = -2.0 / 3.0;
    out[4 * 7 + 4] = 0.0;
    out[4 * 7 + 5] = 2.0 / 3.0;
    out[4 * 7 + 6] = -1.0 / 12.0;
}

} // namespace

struct tension_E4u_1 {

    static constexpr int P = 2;
    static constexpr int R = 5;
    static constexpr int T = 7;
    static constexpr int X = 0;

    real sigma{};
    std::array<real, R * T> cached_coeffs{};

    tension_E4u_1() = default;
    explicit tension_E4u_1(real sigma_in) : sigma{sigma_in}, cached_coeffs{}
    {
        solve_tension_coefficients(sigma, cached_coeffs);
    }

    info query_max() const { return {P, R, T, X}; }
    info query(bcs::type b) const
    {
        switch (b) {
        case bcs::Dirichlet:
            return {P, R - 1, T, 0};
        case bcs::Floating:
            return {P, R, T, 0};
        case bcs::Neumann:
            return {};
        default:
            return {};
        }
    }
    interp_info query_interp() const { return {}; }

    std::span<const real> interp_interior(real, std::span<real> c) const { return c; }

    std::span<const real> interp_wall(int, real, real, std::span<real> c, bool) const
    {
        return c;
    }

    std::span<const real> interior(real h, std::span<real> c) const
    {
        c[0] = 1 / (12 * h);
        c[1] = -2 / (3 * h);
        c[2] = 0;
        c[3] = -c[1];
        c[4] = -c[0];

        return c.subspan(0, 2 * P + 1);
    }

    std::span<const real> nbs(real h,
                              bcs::type b,
                              real psi,
                              bool right,
                              std::span<real> c,
                              std::span<real>) const
    {
        switch (b) {
        case bcs::Floating:
            return nbs_floating(h, psi, c.subspan(0, R * T), right);
        case bcs::Dirichlet:
            return nbs_dirichlet(h, psi, c.subspan(0, (R - 1) * T), right);
        default:
            return c;
        }
    }

    std::span<const real>
    nbs_floating(real h, real, std::span<real> c, bool right) const
    {
        std::copy(cached_coeffs.begin(), cached_coeffs.end(), c.begin());

        for (auto&& v : c) v /= h;
        if (right) {
            for (auto&& v : c) v *= -1;
            std::ranges::reverse(c);
        }

        return c;
    }

    std::span<const real>
    nbs_dirichlet(real h, real, std::span<real> c, bool right) const
    {
        // Dirichlet drops the first (wall) row of the floating block.
        std::copy(cached_coeffs.begin() + T, cached_coeffs.end(), c.begin());

        for (auto&& v : c) v /= h;
        if (right) {
            for (auto&& v : c) v *= -1;
            std::ranges::reverse(c);
        }

        return c;
    }

    void nbs_neumann(real, real, std::span<real>, std::span<real>, bool) const {}
};

stencil make_tension_E4u_1(real sigma) { return tension_E4u_1{sigma}; }

} // namespace ccs::stencils
