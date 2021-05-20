#include "stencil.hpp"

#include <range/v3/algorithm/fill.hpp>
#include <range/v3/algorithm/reverse.hpp>

namespace ccs::stencils
{
struct E2_2 {

    static constexpr int P = 1;
    static constexpr int R = 2;
    static constexpr int T = 4;
    static constexpr int X = 2;

    info query_max() const { return {P, R, T, X}; }
    info query(bcs::type b) const
    {
        switch (b) {
        case bcs::Dirichlet:
            return {P, R - 1, T, 0};
        case bcs::Floating:
            return {P, R, T, 0};
        case bcs::Neumann:
            return {P, R, T, X};
        default:
            return {};
        }
    }

    void interior(real h, std::span<real> c) const
    {
        c[0] = 1 / (h * h);
        c[1] = -2 / (h * h);
        c[2] = 1 / (h * h);
    }

    void nbs(real h,
             bcs::type b,
             real psi,
             bool right,
             std::span<real> c,
             std::span<real> x) const
    {
        switch (b) {
        case bcs::Floating:
            return nbs_floating(h, psi, c.subspan(0, R * T), right);
        case bcs::Dirichlet:
            return nbs_dirichlet(h, psi, c.subspan(0, (R - 1) * T), right);
        case bcs::Neumann:
            return nbs_neumann(h, psi, c.subspan(0, R * T), x.subspan(0, X), right);
        }
    }

    void nbs_floating(real h, real psi, std::span<real> c, bool right) const
    {
        real t6 = psi * psi;
        real t8 = psi * psi * psi;
        real t9 = -1 * t8;
        real t5 = -2 * psi;
        c[0] = psi;
        c[1] = (2 + t5 + -3 * t6 + t9) * 0.5;
        c[2] = -2 + 2 * t6 + t8;
        c[3] = (2 + -1 * t6 + t9) * 0.5;
        c[4] = (2 + 4 * psi) * 1.0 / (2 + 3 * psi + t6);
        c[5] = t5;
        c[6] = (-2 + 4 * t6) * 1.0 / (1 + psi);
        c[7] = (2 + -2 * t6) * 1.0 / (2 + psi);
        for (auto&& v : c) v /= (h * h);

        if (right) ranges::reverse(c);
    }

    void nbs_dirichlet(real h, real psi, std::span<real> c, bool right) const
    {
        real t6 = psi * psi;
        // real t8 = psi * psi * psi;
        // real t9 = -1 * t8;
        real t5 = -2 * psi;
        // c[0] = psi;
        // c[1] = (2 + t5 + -3 * t6 + t9) * 0.5;
        // c[2] = -2 + 2 * t6 + t8;
        // c[3] = (2 + -1 * t6 + t9) * 0.5;
        c[0] = (2 + 4 * psi) * 1.0 / (2 + 3 * psi + t6);
        c[1] = t5;
        c[2] = (-2 + 4 * t6) * 1.0 / (1 + psi);
        c[3] = (2 + -2 * t6) * 1.0 / (2 + psi);
        for (auto&& v : c) v /= (h * h);

        if (right) ranges::reverse(c);
        // nbs_floating(h, psi, c, right);
        // if (right)
        //     ranges::fill(c.subspan((R - 1) * T), real{});
        // else
        //     ranges::fill(c.subspan(0, T), real{});
    }

    void
    nbs_neumann(real h, real psi, std::span<real> c, std::span<real> x, bool right) const
    {
        real t5 = 2 * psi;
        real t9 = psi * psi;
        real t18 = -1 + psi;
        real t23 = psi * psi * psi;
        x[0] = -2;
        x[1] = 2 * t18;
        //
        c[0] = (4 + 8 * psi) * -1 * 1.0 / (2 + 3 * psi + t9);
        c[1] = t5;
        c[2] = (2 + t5 + -4 * t9) * 1.0 / (1 + psi);
        c[3] = 2 * psi * 1.0 / (2 + psi) * t18;
        c[4] = psi;
        c[5] = (-4 + -1 * t23 + t9) * 0.5;
        c[6] = 2 + t23 + -2 * t9;
        c[7] = (2 + -3 * psi + t9) * -0.5 * psi;

        for (auto&& v : c) v /= (h * h);
        for (auto&& v : x) v /= h;

        if (right) {
            ranges::reverse(c);
            ranges::reverse(x);
            for (auto&& v : x) v *= -1;
        }
    }
};

stencil make_E2_2() { return E2_2{}; }

namespace second
{
stencil E2{E2_2{}};
}
} // namespace ccs::stencils
