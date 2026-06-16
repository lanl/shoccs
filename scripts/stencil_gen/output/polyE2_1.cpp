#include "stencil.hpp"

#include <algorithm>

#include <cmath>

namespace ccs::stencils
{
struct polyE2_1 {

    static constexpr int P = 1;
    static constexpr int R = 3;
    static constexpr int T = 4;
    static constexpr int X = 0;

    std::array<real, 6> fa;
    std::array<real, 3> da;
    std::array<real, 4> ia;

    polyE2_1() = default;
    polyE2_1(std::span<const real> fa_,
             std::span<const real> da_,
             std::span<const real> ia_)
    {
        copy_zero_padded(fa_, fa);
        copy_zero_padded(da_, da);
        copy_zero_padded(ia_, ia);
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
    interp_info query_interp() const { return {2, 4}; }

    std::span<const real> interp_interior(real y, std::span<real> c) const
    {
        if (y > 0) {
            c[0] = 1 - y;
            c[1] = y;
        } else {
            c[0] = -y;
            c[1] = y + 1;
        }
        return c.subspan(0, 2);
    }

    std::span<const real>
    interp_wall(int i, real y, real psi, std::span<real> c, bool right) const
    {
        if (right) {
            const real t5 = (1.0 / 2)*y;
            const real t6 = fa[0]*t5 + (1.0 / 2)*ia[0];
            const real t7 = 2*psi;
            const real t8 = 1.0 / (t7 + 2);
            const real t9 = 2*y;
            const real t10 = psi*y;
            const real t11 = t7*y;
            const real t12 = t10 + t9;
            const real t13 = t5 + 1.0 / 2;
            const real t14 = y + 1;
            const real t15 = fa[2]*t5 + (1.0 / 2)*ia[2];
            switch (i) {
            case 0:
                c[0] = fa[1]*t5 + (1.0 / 2)*ia[1] + t6;
                c[1] = t8*(-fa[0]*t11 - fa[0]*t9 - fa[1]*t10 - fa[1]*t9 - ia[0]*t7 - 2*ia[0] - ia[1]*psi - 2*ia[1] - psi * psi - psi - t12);
                c[2] = (1.0 / 2)*psi + t13 + t6;
                c[3] = t8*(fa[1]*y + ia[1] + psi + t14);
                break;
            case 1:
                c[0] = fa[3]*t5 + (1.0 / 2)*ia[3] + t15;
                c[1] = t8*(-fa[2]*t11 - fa[2]*t9 - fa[3]*t10 - fa[3]*t9 - ia[2]*t7 - 2*ia[2] - ia[3]*psi - 2*ia[3] + psi - t12);
                c[2] = t13 + t15;
                c[3] = t8*(fa[3]*y + ia[3] + t14);
                break;
            }
        } else {
            const real t5 = 2*psi;
            const real t6 = 1.0 / (t5 + 2);
            const real t7 = fa[1]*y;
            const real t8 = 1 - y;
            const real t9 = (1.0 / 2)*y;
            const real t10 = fa[0]*t9 + (1.0 / 2)*ia[0];
            const real t11 = 1.0 / 2 - t9;
            const real t12 = 2*y;
            const real t13 = t5*y;
            const real t14 = -psi*y - 2*y;
            const real t15 = fa[2]*t9 + (1.0 / 2)*ia[2];
            switch (i) {
            case 0:
                c[0] = t6*(ia[1] + psi + t7 + t8);
                c[1] = (1.0 / 2)*psi + t10 + t11;
                c[2] = t6*(-fa[0]*t12 - fa[0]*t13 - ia[0]*t5 - 2*ia[0] - ia[1]*psi - 2*ia[1] - psi * psi - psi*t7 - psi - t14 - 2*t7);
                c[3] = (1.0 / 2)*ia[1] + t10 + (1.0 / 2)*t7;
                break;
            case 1:
                c[0] = t6*(fa[3]*y + ia[3] + t8);
                c[1] = t11 + t15;
                c[2] = t6*(-fa[2]*t12 - fa[2]*t13 - fa[3]*psi*y - fa[3]*t12 - ia[2]*t5 - 2*ia[2] - ia[3]*psi - 2*ia[3] + psi - t14);
                c[3] = fa[3]*t9 + (1.0 / 2)*ia[3] + t15;
                break;
            }
        }
        return c.subspan(0, 4);
    }

    std::span<const real> interior(real h, std::span<real> c) const
    {
        c = c.subspan(0, 2 * 1 + 1);
        c[0] = -0.5;
        c[1] = 0;
        c[2] = 0.5;
        for (auto&& v : c) v /= h;
        return c;
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
            return nbs_floating(h, psi, c, right);
        case bcs::Dirichlet:
            return nbs_dirichlet(h, psi, c, right);
        default:
            return c;
        }
    }

    std::span<const real>
    nbs_floating(real h, real psi, std::span<real> c, bool right) const
    {
        c = c.subspan(0, R * T);

        real t5 = 2*psi;
        real t6 = 1.0 / (t5 + 2);
        real t7 = (1.0 / 2)*fa[0];
        real t8 = -psi - 2;
        real t9 = (1.0 / 2)*fa[2];
        real t10 = (1.0 / 2)*fa[4];

        c[0] = t6*(fa[1] - 1);
        c[1] = t7 - 1.0 / 2;
        c[2] = t6*(-fa[0]*t5 - 2*fa[0] - fa[1]*psi - 2*fa[1] - t8);
        c[3] = (1.0 / 2)*fa[1] + t7;
        c[4] = t6*(fa[3] - 1);
        c[5] = t9 - 1.0 / 2;
        c[6] = t6*(-fa[2]*t5 - 2*fa[2] - fa[3]*psi - 2*fa[3] - t8);
        c[7] = (1.0 / 2)*fa[3] + t9;
        c[8] = t6*(fa[5] - 1);
        c[9] = t10 - 1.0 / 2;
        c[10] = t6*(-fa[4]*t5 - 2*fa[4] - fa[5]*psi - 2*fa[5] - t8);
        c[11] = (1.0 / 2)*fa[5] + t10;

        for (auto&& v : c) v /= h;
        if (right) {
            for (auto&& v : c) v *= -1;
            std::ranges::reverse(c);
        }

        return c;
    }

    std::span<const real>
    nbs_dirichlet(real h, real psi, std::span<real> c, bool right) const
    {
        c = c.subspan(0, (R - 1) * T);

        real t5 = 2*psi;
        real t6 = 1.0 / (t5 + 2);
        real t7 = 4*psi;
        real t8 = 6*da[2];
        real t9 = t8 + 2;
        real t10 = 1.0 / (da[2]*t7 + t7 + t9);
        real t11 = 3*da[0];
        real t12 = da[0]*t5;
        real t13 = da[1]*t5;
        real t14 = da[1]*t11 + da[1]*t12 + da[1] + t13;
        real t15 = psi * psi;
        real t16 = 4*t15;
        real t17 = 6*psi;
        real t18 = da[2]*psi;
        real t19 = 2*da[2]*t15;
        real t20 = 3*da[2];
        real t21 = da[0]*da[1];
        real t22 = 2*da[1] - 2;
        real t23 = (1.0 / 2)*da[1];

        c[0] = t6*(da[0] - 1);
        c[1] = t10*(-t11 - t12 + t14 - t5 - 1);
        c[2] = (5*da[0]*psi + 2*da[0]*t15 - 7*da[0]*t18 - da[0]*t19 - da[0]*t8 + 4*da[0] - da[1]*t16 - da[1]*t17 - psi*t20 - 10*psi*t21 + 5*psi + 2*t15 - t16*t21 - t19 - 6*t21 - t22)/(da[2]*t16 + t16 + t17 + 10*t18 + t9);
        c[3] = t10*(-2*da[0] + da[2]*t11 + da[2]*t12 + da[2]*t5 + t14 + t20);
        c[4] = t6*(da[2] - 1);
        c[5] = t23 - 1.0 / 2;
        c[6] = t6*(-2*da[2] + psi - t13 - t18 - t22);
        c[7] = (1.0 / 2)*da[2] + t23;

        for (auto&& v : c) v /= h;
        if (right) {
            for (auto&& v : c) v *= -1;
            std::ranges::reverse(c);
        }

        return c;
    }

    std::span<const real>
    nbs_neumann(real, real, std::span<real> c, std::span<real>, bool) const
    {
        return c;
    }
};

stencil make_polyE2_1(std::span<const real> fa,
                      std::span<const real> da,
                      std::span<const real> ia)
{
    return polyE2_1{fa, da, ia};
}

} // namespace ccs::stencils
