#include "stencil.hpp"

#include <algorithm>
#include <stdexcept>

#include <cmath>

namespace ccs::stencils
{
struct E4_1 {

    static constexpr int P = 2;
    static constexpr int R = 5;
    static constexpr int T = 7;
    static constexpr int X = 0;

    // Singularity constraints for the conservative E4_1 stencil:
    //   - psi must be in the open interval (0, 1). Coefficients have poles at
    //     psi=0 (nbs_floating divides by psi) and psi=1
    //     (nbs_floating, nbs_dirichlet divide by (psi - 1)).
    //   - alpha[1] must be >= 197/288 ≈ 0.684 to avoid an interior singularity
    //     in the polynomial denominator 1728*alpha[1] + 1584*alpha[1]*psi +
    //     864*alpha[1]*psi^2 + 144*alpha[1]*psi^3 + 12*psi^6 + 162*psi^5 +
    //     1464*psi^4 + 5617*psi^3 + 8070*psi^2 + 1721*psi - 1182.
    //     D(0) = 1728*alpha[1] - 1182 = 0 when alpha[1] = 197/288.
    //     D'(psi) > 0 for all psi >= 0 and alpha[1] > 0, so D is strictly
    //     increasing; D(0) >= 0 ensures D(psi) > 0 for all psi in (0, 1).
    //
    // alpha[0]: boundary shape parameter (free)
    // alpha[1]: quadrature weight parameter (must be >= 197/288, see above)
    std::array<real, 2> alpha;

    E4_1() = default;
    E4_1(std::span<const real> a)
    {
        copy_zero_padded(a, alpha);
        if (alpha[1] < 197.0 / 288.0)
            throw std::invalid_argument(
                "E4_1: alpha[1] must be >= 197/288 ≈ 0.684 to avoid interior "
                "denominator singularity");
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
        c = c.subspan(0, 2 * 2 + 1);
        c[0] = 0.083333333333333329;
        c[1] = -0.66666666666666663;
        c[2] = 0;
        c[3] = 0.66666666666666663;
        c[4] = -0.083333333333333329;
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

        constexpr real psi_eps = 1e-4;
        psi = std::clamp(psi, psi_eps, 1.0 - psi_eps);

        real t5 = alpha[1]*psi;
        real t6 = psi * psi;
        real t7 = psi * psi * psi;
        real t8 = std::pow(psi, 4);
        real t9 = std::pow(psi, 5);
        real t10 = std::pow(psi, 6);
        real t11 = std::pow(psi, 7);
        real t12 = alpha[1]*t6;
        real t13 = alpha[1]*t7;
        real t14 = alpha[1]*t8;
        real t15 = alpha[1]*t9;
        real t16 = alpha[1]*t10;
        real t17 = alpha[1]*t11;
        real t18 = alpha[1]*std::pow(psi, 8);
        real t19 = alpha[1] * alpha[1];
        real t20 = psi*t19;
        real t21 = alpha[0]*t12;
        real t22 = alpha[0]*t13;
        real t23 = alpha[0]*t14;
        real t24 = alpha[0]*t15;
        real t25 = alpha[0]*t16;
        real t26 = t19*t6;
        real t27 = 99360*t19;
        real t28 = t19*t9;
        real t29 = alpha[0]*t26;
        real t30 = alpha[0]*t19;
        real t31 = 17280*t30;
        real t32 = 2484*alpha[0]*t17 + 216*alpha[0]*t18 + 3456*alpha[0]*t28 + 10368*alpha[1] + 13872*psi - 414*t10 - 36*t11 + 314677*t12 + 1282709*t13 - 182437*t14 - 496397*t15 - 169230*t16 - 21558*t17 - 1716*t18 + 276480*t20 + 20394*t21 - 94806*t22 + 14166*t23 + 41670*t24 + 15876*t25 + 218592*t26 - t27*t7 - t27*t8 - 19872*t28 - 38016*t29 + t31*t7 + t31*t8 - 184800*t5 + 33753*t6 + 9060*t7 - 6339*t8 - 2988*t9 - 7092;
        real t33 = 1.0 / (alpha[1]);
        real t34 = 1.0 / (psi - 1);
        real t35 = 1.0 / (1728*alpha[1] + 1721*psi + 12*t10 + 864*t12 + 144*t13 + 1584*t5 + 8070*t6 + 5617*t7 + 1464*t8 + 162*t9 - 1182);
        real t36 = t33*t34*t35;
        real t37 = t32*t36;
        real t38 = (1.0 / 6)*t37;
        real t39 = 1.0 / (psi);
        real t40 = t38*t39;
        real t41 = -t40;
        real t42 = -2.0 / 3*t32*t33*t34*t35*t39;
        real t43 = t37*t6;
        real t44 = psi*t37;
        real t45 = t37*t7;
        real t46 = (1.0 / 36)*t45 + (11.0 / 36)*t8;
        real t47 = (7.0 / 12)*t6;
        real t48 = (1.0 / 12)*t45 + (11.0 / 12)*t8;
        real t49 = (5.0 / 36)*t6;
        real t50 = alpha[0]*psi;
        real t51 = 6*alpha[0];
        real t52 = 168*alpha[0]*alpha[1] - 832*alpha[1]*t50 + 342*alpha[1] - 288*psi - 10194*t12 + 4350*t13 + 4671*t14 + 1032*t15 + 33*t16 + t17*t51 + 3*t17 - 756*t19*t50 + 330*t19*t7 + 66*t19*t8 + 5034*t20 - 770*t21 + 274*t22 + 752*t23 + 336*t24 + 66*t25 + 330*t26 + 540*t29 + 540*t30*t7 + 108*t30*t8 - 432*t30 - 5565*t5 - 18*t6 + 72*t7 + 18*t8 - 216;
        real t53 = t36*t52;
        real t54 = 4*t53;
        real t55 = psi*t54;
        real t56 = (1.0 / 3)*psi;
        real t57 = t56 - 1.0 / 3;
        real t58 = t53*t8;
        real t59 = -2.0 / 3*t58;
        real t60 = 16*t53;
        real t61 = psi*t53;
        real t62 = (1.0 / 18)*t8;
        real t63 = -13.0 / 9*psi + (17.0 / 18)*t6 + t62 + (4.0 / 9)*t7;
        real t64 = 24*t53;
        real t65 = t53*t6;
        real t66 = t53*t7;
        real t67 = -2*t58;
        real t68 = (1.0 / 6)*t8;
        real t69 = -19.0 / 6*psi + (11.0 / 6)*t6 + t68 + (7.0 / 6)*t7;
        real t70 = -7.0 / 3*psi + (7.0 / 6)*t6 + t68 + t7;
        real t71 = -11.0 / 18*psi + (5.0 / 18)*t6 + t62 + (5.0 / 18)*t7;
        real t72 = -alpha[0];
        real t73 = -4*alpha[0];
        real t74 = alpha[0]*t7;
        real t75 = alpha[0]*t6;
        real t76 = alpha[0]*t68 + (1.0 / 36)*t8;
        real t77 = (1.0 / 2)*alpha[0]*t8 + (1.0 / 12)*t8;
        real t78 = 6*t50;
        real t79 = psi*t64;
        real t80 = -1.0 / 6*t32*t33*t34*t35;
        real t81 = 2*psi;
        real t82 = (4.0 / 3)*t37;
        real t83 = 8*psi;
        real t84 = t51*t6 + t6*t64 - 149*t6;

        c[0] = (11.0 / 6)*psi + t38 + t41 - 11.0 / 6;
        c[1] = psi*(-t40 - 11.0 / 6);
        c[2] = (143.0 / 18)*psi + (13.0 / 18)*t32*t33*t34*t35 - t42 - 2.0 / 9*t43 - 17.0 / 36*t44 - t46 - 187.0 / 36*t6 - 22.0 / 9*t7 + 3;
        c[3] = -209.0 / 12*psi - t37*t39 + t37*t47 - 19.0 / 12*t37 + (11.0 / 12)*t44 + t48 + (121.0 / 12)*t6 + (77.0 / 12)*t7 - 3.0 / 2;
        c[4] = (77.0 / 6)*psi + (7.0 / 6)*t32*t33*t34*t35 - t42 - 1.0 / 2*t43 - 7.0 / 12*t44 - t48 - 77.0 / 12*t6 - 11.0 / 2*t7 + 1.0 / 3;
        c[5] = -121.0 / 36*psi + t37*t49 - 11.0 / 36*t37 + t41 + (5.0 / 36)*t44 + t46 + (55.0 / 36)*t6 + (55.0 / 36)*t7;
        c[6] = 0;
        c[7] = t54 - t55 + t57;
        c[8] = psi*(t54 - 1.0 / 3);
        c[9] = (34.0 / 3)*t33*t34*t35*t52*t6 + (16.0 / 3)*t33*t34*t35*t52*t7 - t59 - t60 - 52.0 / 3*t61 - t63 - 1.0 / 2;
        c[10] = 38*t61 + t64 - 22*t65 - 14*t66 + t67 + t69 + 1;
        c[11] = 14*t33*t34*t35*t52*t6 + 12*t33*t34*t35*t52*t7 - t60 - 28*t61 - t67 - t70 - 1.0 / 6;
        c[12] = t54 + t59 + (22.0 / 3)*t61 - 10.0 / 3*t65 - 10.0 / 3*t66 + t71;
        c[13] = 0;
        c[14] = -1.0 / 6*psi - t50 - t72 + 1.0 / 6;
        c[15] = psi*(alpha[0] + 1.0 / 6);
        c[16] = -13.0 / 18*psi - 13.0 / 3*t50 + (17.0 / 36)*t6 + (2.0 / 9)*t7 + t73 + (4.0 / 3)*t74 + (17.0 / 6)*t75 + t76 - 1;
        c[17] = (19.0 / 2)*alpha[0]*psi + (19.0 / 12)*psi + t51 - 11.0 / 12*t6 - 7.0 / 12*t7 - 7.0 / 2*t74 - 11.0 / 2*t75 - t77 + 1.0 / 2;
        c[18] = -7.0 / 6*psi + t47 - 7*t50 + (1.0 / 2)*t7 + t73 + 3*t74 + (7.0 / 2)*t75 + t77 + 1.0 / 3;
        c[19] = (11.0 / 6)*alpha[0]*psi + (11.0 / 36)*psi - t49 - 5.0 / 36*t7 - t72 - 5.0 / 6*t74 - 5.0 / 6*t75 - t76;
        c[20] = 0;
        c[21] = t57;
        c[22] = -t56;
        c[23] = 3.0 / 2 - t63;
        c[24] = t69 - 3;
        c[25] = 11.0 / 6 - t70;
        c[26] = t71;
        c[27] = 0;
        c[28] = (-149*psi - t51 - t64 + t78 + t79 - 11)/(11*psi + 6*t6 + t7 + 6);
        c[29] = (149.0 / 6)*psi - t50 - t55 - t80;
        c[30] = (8*alpha[0] - psi*t82 + 3*psi + 2*t50 + t53*t83 + 32*t53 - t82 + t84 + 14)/(t81 + 2);
        c[31] = (-24*alpha[0] + 2*psi*t32*t33*t34*t35 + 4*t32*t33*t34*t35 - 96*t53 - t78 - t79 - t83 - t84 - 38)/(t81 + 4);
        c[32] = (72*alpha[0] - 4*psi*t37 + 15*psi - 12*t37 + 18*t50 + 288*t53 + 72*t61 + t84 + 78)/(6*psi + 18);
        c[33] = -alpha[0] - t54 - t80;
        c[34] = 0;

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

        constexpr real psi_eps = 1e-4;
        psi = std::clamp(psi, psi_eps, 1.0 - psi_eps);

        real t5 = alpha[0]*alpha[1];
        real t6 = alpha[1]*psi;
        real t7 = psi * psi;
        real t8 = psi * psi * psi;
        real t9 = std::pow(psi, 4);
        real t10 = alpha[0]*psi;
        real t11 = alpha[1] * alpha[1];
        real t12 = alpha[0]*t11;
        real t13 = alpha[1]*t7;
        real t14 = alpha[1]*t8;
        real t15 = alpha[1]*t9;
        real t16 = std::pow(psi, 5);
        real t17 = alpha[1]*t16;
        real t18 = std::pow(psi, 6);
        real t19 = alpha[1]*t18;
        real t20 = std::pow(psi, 7);
        real t21 = alpha[1]*t20;
        real t22 = psi*t11;
        real t23 = alpha[0]*t13;
        real t24 = alpha[0]*t14;
        real t25 = t5*t9;
        real t26 = t16*t5;
        real t27 = alpha[0]*t19;
        real t28 = 6*alpha[0];
        real t29 = 330*t11;
        real t30 = t11*t9;
        real t31 = 540*t12;
        real t32 = t12*t9;
        real t33 = -832*alpha[1]*t10 + 342*alpha[1] - 288*psi - 756*t10*t11 - 432*t12 - 10194*t13 + 4350*t14 + 4671*t15 + 1032*t17 + 33*t19 + t21*t28 + 3*t21 + 5034*t22 - 770*t23 + 274*t24 + 752*t25 + 336*t26 + 66*t27 + t29*t7 + t29*t8 + 66*t30 + t31*t7 + t31*t8 + 108*t32 + 168*t5 - 5565*t6 - 18*t7 + 72*t8 + 18*t9 - 216;
        real t34 = 1.0 / (alpha[1]);
        real t35 = 1.0 / (psi - 1);
        real t36 = 1.0 / (1728*alpha[1] + 1721*psi + 864*t13 + 144*t14 + 162*t16 + 12*t18 + 1584*t6 + 8070*t7 + 5617*t8 + 1464*t9 - 1182);
        real t37 = t34*t35*t36;
        real t38 = t33*t37;
        real t39 = 4*t38;
        real t40 = psi*t39;
        real t41 = (1.0 / 3)*psi;
        real t42 = t41 - 1.0 / 3;
        real t43 = t38*t9;
        real t44 = -2.0 / 3*t43;
        real t45 = 16*t38;
        real t46 = psi*t38;
        real t47 = (1.0 / 18)*t9;
        real t48 = -13.0 / 9*psi + t47 + (17.0 / 18)*t7 + (4.0 / 9)*t8;
        real t49 = 24*t38;
        real t50 = t38*t7;
        real t51 = t38*t8;
        real t52 = -2*t43;
        real t53 = (1.0 / 6)*t9;
        real t54 = -19.0 / 6*psi + t53 + (11.0 / 6)*t7 + (7.0 / 6)*t8;
        real t55 = -7.0 / 3*psi + t53 + (7.0 / 6)*t7 + t8;
        real t56 = -11.0 / 18*psi + t47 + (5.0 / 18)*t7 + (5.0 / 18)*t8;
        real t57 = -alpha[0];
        real t58 = -4*alpha[0];
        real t59 = alpha[0]*t8;
        real t60 = alpha[0]*t7;
        real t61 = alpha[0]*t53 + (1.0 / 36)*t9;
        real t62 = (1.0 / 2)*alpha[0]*t9 + (1.0 / 12)*t9;
        real t63 = 6*t10;
        real t64 = psi*t49;
        real t65 = std::pow(psi, 8);
        real t66 = 2484*alpha[0]*t21 - 1716*alpha[1]*t65 + 10368*alpha[1] + 13872*psi - 19872*t11*t16 + 218592*t11*t7 - 99360*t11*t8 + 3456*t12*t16 - 38016*t12*t7 + 17280*t12*t8 + 314677*t13 + 1282709*t14 - 182437*t15 - 2988*t16 - 496397*t17 - 414*t18 - 169230*t19 - 36*t20 - 21558*t21 + 276480*t22 + 20394*t23 - 94806*t24 + 14166*t25 + 41670*t26 + 15876*t27 - 99360*t30 + 17280*t32 + 216*t5*t65 - 184800*t6 + 33753*t7 + 9060*t8 - 6339*t9 - 7092;
        real t67 = -1.0 / 6*t34*t35*t36*t66;
        real t68 = 2*psi;
        real t69 = t37*t66;
        real t70 = (4.0 / 3)*t69;
        real t71 = 8*psi;
        real t72 = t28*t7 + t49*t7 - 149*t7;

        c[0] = t39 - t40 + t42;
        c[1] = psi*(t39 - 1.0 / 3);
        c[2] = (34.0 / 3)*t33*t34*t35*t36*t7 + (16.0 / 3)*t33*t34*t35*t36*t8 - t44 - t45 - 52.0 / 3*t46 - t48 - 1.0 / 2;
        c[3] = 38*t46 + t49 - 22*t50 - 14*t51 + t52 + t54 + 1;
        c[4] = 14*t33*t34*t35*t36*t7 + 12*t33*t34*t35*t36*t8 - t45 - 28*t46 - t52 - t55 - 1.0 / 6;
        c[5] = t39 + t44 + (22.0 / 3)*t46 - 10.0 / 3*t50 - 10.0 / 3*t51 + t56;
        c[6] = 0;
        c[7] = -1.0 / 6*psi - t10 - t57 + 1.0 / 6;
        c[8] = psi*(alpha[0] + 1.0 / 6);
        c[9] = -13.0 / 18*psi - 13.0 / 3*t10 + t58 + (4.0 / 3)*t59 + (17.0 / 6)*t60 + t61 + (17.0 / 36)*t7 + (2.0 / 9)*t8 - 1;
        c[10] = (19.0 / 2)*alpha[0]*psi + (19.0 / 12)*psi + t28 - 7.0 / 2*t59 - 11.0 / 2*t60 - t62 - 11.0 / 12*t7 - 7.0 / 12*t8 + 1.0 / 2;
        c[11] = -7.0 / 6*psi - 7*t10 + t58 + 3*t59 + (7.0 / 2)*t60 + t62 + (7.0 / 12)*t7 + (1.0 / 2)*t8 + 1.0 / 3;
        c[12] = (11.0 / 6)*alpha[0]*psi + (11.0 / 36)*psi - t57 - 5.0 / 6*t59 - 5.0 / 6*t60 - t61 - 5.0 / 36*t7 - 5.0 / 36*t8;
        c[13] = 0;
        c[14] = t42;
        c[15] = -t41;
        c[16] = 3.0 / 2 - t48;
        c[17] = t54 - 3;
        c[18] = 11.0 / 6 - t55;
        c[19] = t56;
        c[20] = 0;
        c[21] = (-149*psi - t28 - t49 + t63 + t64 - 11)/(11*psi + 6*t7 + t8 + 6);
        c[22] = (149.0 / 6)*psi - t10 - t40 - t67;
        c[23] = (8*alpha[0] - psi*t70 + 3*psi + 2*t10 + t38*t71 + 32*t38 - t70 + t72 + 14)/(t68 + 2);
        c[24] = (-24*alpha[0] + 2*psi*t34*t35*t36*t66 + 4*t34*t35*t36*t66 - 96*t38 - t63 - t64 - t71 - t72 - 38)/(t68 + 4);
        c[25] = (72*alpha[0] - 4*psi*t69 + 15*psi + 18*t10 + 288*t38 + 72*t46 - 12*t69 + t72 + 78)/(6*psi + 18);
        c[26] = -alpha[0] - t39 - t67;
        c[27] = 0;

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

stencil make_E4_1(std::span<const real> alpha) { return E4_1{alpha}; }

} // namespace ccs::stencils
