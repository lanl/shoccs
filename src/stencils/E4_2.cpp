#include "stencil.hpp"

#include <range/v3/algorithm/fill.hpp>
#include <range/v3/algorithm/reverse.hpp>

#include <cassert>

namespace ccs::stencils
{
struct E4_2 {

    static constexpr int P = 2;
    static constexpr int R = 3;
    static constexpr int T = 5;
    static constexpr int X = 3;

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

    interp_info query_interp() const { return {P, T}; }

    void interior(real h, std::span<real> c) const
    {
        c[0] = -1 / (12. * (h * h));
        c[1] = 4 / (3. * (h * h));
        c[2] = -5 / (2. * (h * h));
        c[3] = 4 / (3. * (h * h));
        c[4] = -1 / (12. * (h * h));
    }

    std::span<const real> interp_interior(real y, std::span<real> c) const
    {
        real y2 = y * y;
        real y3 = y * y * y;
        real y4 = y2 * y2;

        c[0] = (y * (2 - y - 2 * y2 + y3)) / 24.;
        c[1] = (y * (-4 + 4 * y + y2 - y3)) / 6.;
        c[2] = (4 - 5 * y2 + y4) / 4.;
        c[3] = -(y * (-4 - 4 * y + y2 + y3)) / 6.;
        c[4] = (y * (-2 - y + 2 * y2 + y3)) / 24.;
        return c.subspan(0, 2 * P + 1);
    }

    std::span<const real>
    interp_wall(int i, real y, real psi, std::span<real> c, bool right) const
    {
        real y2 = y * y;
        real y3 = y2 * y;
        real psi2 = psi * psi;
        real psi3 = psi2 * psi;
        real psi4 = psi2 * psi2;

        if (right) {
            switch (i) {
            case 0:

                c[0] = ((-1 + psi) *
                        (6 * y * (2 + 3 * y + y2) + 6 * psi * (2 + 8 * y + 6 * y2 + y3) +
                         psi3 * (6 + 11 * y + 6 * y2 + y3) +
                         2 * psi2 * (9 + 22 * y + 12 * y2 + 2 * y3))) /
                       36.;
                c[1] = (6 * y * (3 + 4 * y + y2) + 6 * psi * (3 + 8 * y + 3 * y2) -
                        3 * psi2 * (-2 + 5 * y + 6 * y2 + y3) -
                        psi4 * (6 + 11 * y + 6 * y2 + y3) -
                        2 * psi3 * (9 + 22 * y + 12 * y2 + 2 * y3)) /
                       12.;
                c[2] = (-6 * y * (6 + 5 * y + y2) - 6 * psi * (6 + 10 * y + 3 * y2) +
                        6 * psi2 * (1 + 8 * y + 6 * y2 + y3) +
                        psi4 * (6 + 11 * y + 6 * y2 + y3) +
                        psi3 * (24 + 55 * y + 30 * y2 + 5 * y3)) /
                       12.;
                c[3] =
                    (-6 * psi * (-5 - y + 3 * y2 + y3) -
                     6 * psi3 * (5 + 11 * y + 6 * y2 + y3) +
                     6 * (6 + 11 * y + 6 * y2 + y3) - psi4 * (6 + 11 * y + 6 * y2 + y3) -
                     psi2 * (30 + 103 * y + 66 * y2 + 11 * y3)) /
                    36.;
                c[4] = (psi * (6 + 11 * y + 6 * y2 + y3)) / 6.;
                break;
            case 1:
                c[0] = ((-1 + psi) * (psi * (-1 + y) + y) * (2 + 3 * y + y2)) /
                       (2. * (3 + psi));
                c[1] = -((1 + y) * (2 * psi * (3 + y) - 2 * y * (3 + y) +
                                    3 * psi2 * (-2 + y + y2))) /
                       (2. * (2 + psi));
                c[2] = ((2 + y) * (psi * (3 + y) - y * (3 + y) + 3 * psi2 * (-1 + y2))) /
                       (2. * (1 + psi));
                c[3] = (psi * (2 + y - 2 * y2 - y3)) / 2.;
                c[4] = ((3 + 3 * psi * (-1 + y) + y) * (2 + 3 * y + y2)) /
                       (6 + 11 * psi + 6 * psi2 + psi3);
                break;
            default:
                c[0] = (y * (-24 * (-1 + y2) + 2 * psi2 * (-2 - y + 2 * y2 + y3) +
                             3 * psi3 * (-2 - y + 2 * y2 + y3) +
                             psi4 * (-2 - y + 2 * y2 + y3))) /
                       144.;
                c[1] = (y * (24 * (-2 + y + y2) + psi4 * (2 + y - 2 * y2 - y3) -
                             3 * psi2 * (-2 - y + 2 * y2 + y3) -
                             4 * psi3 * (-2 - y + 2 * y2 + y3))) /
                       48.;
                c[2] = ((-24 + 6 * psi2 * y + 5 * psi3 * y + psi4 * y) *
                        (-2 - y + 2 * y2 + y3)) /
                       48.;
                c[3] = -(y * (-24 * (2 + 3 * y + y2) + 6 * psi * (-2 - y + 2 * y2 + y3) +
                              11 * psi2 * (-2 - y + 2 * y2 + y3) +
                              6 * psi3 * (-2 - y + 2 * y2 + y3) +
                              psi4 * (-2 - y + 2 * y2 + y3))) /
                       144.;
                c[4] = (psi * y * (-2 - y + 2 * y2 + y3)) / 24.;
            }
        } else {
            switch (i) {
            case 0:
                c[0] = -(psi * (-6 + 11 * y - 6 * y2 + y3)) / 6.;
                c[1] = (-6 * (-6 + 11 * y - 6 * y2 + y3) +
                        psi4 * (-6 + 11 * y - 6 * y2 + y3) +
                        6 * psi3 * (-5 + 11 * y - 6 * y2 + y3) +
                        6 * psi * (5 - y - 3 * y2 + y3) +
                        psi2 * (-30 + 103 * y - 66 * y2 + 11 * y3)) /
                       36.;
                c[2] = (6 * y * (6 - 5 * y + y2) - 6 * psi * (6 - 10 * y + 3 * y2) +
                        psi3 * (24 - 55 * y + 30 * y2 - 5 * y3) -
                        6 * psi2 * (-1 + 8 * y - 6 * y2 + y3) -
                        psi4 * (-6 + 11 * y - 6 * y2 + y3)) /
                       12.;
                c[3] = (-6 * y * (3 - 4 * y + y2) + 6 * psi * (3 - 8 * y + 3 * y2) +
                        3 * psi2 * (2 + 5 * y - 6 * y2 + y3) +
                        psi4 * (-6 + 11 * y - 6 * y2 + y3) +
                        2 * psi3 * (-9 + 22 * y - 12 * y2 + 2 * y3)) /
                       12.;
                c[4] = -((-1 + psi) * (6 * y * (2 - 3 * y + y2) +
                                       6 * psi * (-2 + 8 * y - 6 * y2 + y3) +
                                       psi3 * (-6 + 11 * y - 6 * y2 + y3) +
                                       2 * psi2 * (-9 + 22 * y - 12 * y2 + 2 * y3))) /
                       36.;
                break;
            case 1:
                c[0] = -(((2 - 3 * y + y2) * (-3 + y + 3 * psi * (1 + y))) /
                         ((1 + psi) * (2 + psi) * (3 + psi)));
                c[1] = (psi * (2 - y - 2 * y2 + y3)) / 2.;
                c[2] = -((-2 + y) *
                         (-(psi * (-3 + y)) - (-3 + y) * y + 3 * psi2 * (-1 + y2))) /
                       (2. * (1 + psi));
                c[3] = ((-1 + y) * (-2 * psi * (-3 + y) - 2 * (-3 + y) * y +
                                    3 * psi2 * (-2 - y + y2))) /
                       (2. * (2 + psi));
                c[4] = -((-1 + psi) * (psi + y + psi * y) * (2 - 3 * y + y2)) /
                       (2. * (3 + psi));
                break;
            default:
                c[0] = (psi * y * (2 - y - 2 * y2 + y3)) / 24.;
                c[1] = -(y * (24 * (2 - 3 * y + y2) + 6 * psi * (2 - y - 2 * y2 + y3) +
                              11 * psi2 * (2 - y - 2 * y2 + y3) +
                              6 * psi3 * (2 - y - 2 * y2 + y3) +
                              psi4 * (2 - y - 2 * y2 + y3))) /
                       144.;
                c[2] = ((24 + 6 * psi2 * y + 5 * psi3 * y + psi4 * y) *
                        (2 - y - 2 * y2 + y3)) /
                       48.;
                c[3] = (y * (24 * (2 + y - y2) + psi4 * (-2 + y + 2 * y2 - y3) -
                             3 * psi2 * (2 - y - 2 * y2 + y3) -
                             4 * psi3 * (2 - y - 2 * y2 + y3))) /
                       48.;
                c[4] = (y * (24 * (-1 + y2) + 2 * psi2 * (2 - y - 2 * y2 + y3) +
                             3 * psi3 * (2 - y - 2 * y2 + y3) +
                             psi4 * (2 - y - 2 * y2 + y3))) /
                       144.;
            }
        }
        return c.subspan(0, T);
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
        double t6 = -3 * psi;
        double t7 = psi * psi;
        double t9 = t7 * psi;
        double t11 = t7 * t7;
        double t12 = -t11;
        double t18 = 3 * psi;
        double t33 = -1 + psi;
        double t5 = 2 * psi;
        double t31 = 6 * psi;
        c[0] = t5;
        c[1] = (6 + t12 + t6 - 11 * t7 - 6 * t9) / 3.;
        c[2] = -5 + t11 + t6 + 6 * t7 + 5 * t9;
        c[3] = 4 + t12 + t18 - 3 * t7 - 4 * t9;
        c[4] = (-3 + t11 + t6 + 2 * t7 + 3 * t9) / 3.;
        c[5] = 12 / (6 + 5 * psi + t7);
        c[6] = -2 * psi;
        c[7] = -5 + t31;
        c[8] = (-2 * (4 + t18) * t33) / (2 + psi);
        c[9] = (t33 * (3 + t5)) / (3 + psi);
        c[10] = -psi / 12.;
        c[11] = (72 + t11 + t31 + 11 * t7 + 6 * t9) / 72.;
        c[12] = (-48 + t12 - 6 * t7 - 5 * t9) / 24.;
        c[13] = (24 + t11 + 3 * t7 + 4 * t9) / 24.;
        c[14] = -(t7 * (2 + t18 + t7)) / 72.;

        for (auto&& v : c) v /= (h * h);

        if (right) ranges::reverse(c);
    }

    void nbs_dirichlet(real h, real psi, std::span<real> c, bool right) const
    {
        double t7 = psi * psi;
        double t9 = t7 * psi;
        double t11 = t7 * t7;
        double t12 = -t11;
        double t18 = 3 * psi;
        double t33 = -1 + psi;
        double t5 = 2 * psi;
        double t31 = 6 * psi;

        c[0] = 12 / (6 + 5 * psi + t7);
        c[1] = -2 * psi;
        c[2] = -5 + t31;
        c[3] = (-2 * (4 + t18) * t33) / (2 + psi);
        c[4] = (t33 * (3 + t5)) / (3 + psi);
        c[5] = -psi / 12.;
        c[6] = (72 + t11 + t31 + 11 * t7 + 6 * t9) / 72.;
        c[7] = (-48 + t12 - 6 * t7 - 5 * t9) / 24.;
        c[8] = (24 + t11 + 3 * t7 + 4 * t9) / 24.;
        c[9] = -(t7 * (2 + t18 + t7)) / 72.;
        for (auto&& v : c) v /= (h * h);

        if (right) ranges::reverse(c);
    }

    void
    nbs_neumann(real h, real psi, std::span<real> c, std::span<real> x, bool right) const
    {
        double t7 = psi * psi;
        double t10 = 11 * psi;
        double t28 = -1 + psi;
        double t12 = t7 * psi;
        double t36 = t7 * t7;
        double t37 = -t36;
        double t51 = -6 * t7;
        double t8 = 3 * t7;

        x[0] = -3;
        x[1] = 3 * t28;
        x[2] = 0;

        c[0] = (-3 * (7 + 18 * psi + t8)) / (6 + t10 + t12 + 6 * t7);
        c[1] = 4 * psi;
        c[2] = (8 + t10 - 21 * t7) / (2 + 2 * psi);
        c[3] = (-1 - 8 * psi + 9 * t7) / (2 + psi);
        c[4] = (-5 * psi * t28) / (2. * (3 + psi));

        c[5] = psi;
        c[6] = (-21 - 9 * psi + 3 * t12 + t37 + 16 * t7) / 6.;
        c[7] = (8 + 12 * psi - 4 * t12 + t36 - 15 * t7) / 2.;
        c[8] = (-1 - 15 * psi + 5 * t12 + t37 + 12 * t7) / 2.;
        c[9] = (psi * (12 - 7 * psi + t12 + t51)) / 6.;

        c[10] = -psi / 12.;
        c[11] = (72 + 6 * psi + 6 * t12 + t36 + 11 * t7) / 72.;
        c[12] = (-48 - 5 * t12 + t37 + t51) / 24.;
        c[13] = (24 + 4 * t12 + t36 + t8) / 24.;
        c[14] = -(t7 * (2 + 3 * psi + t7)) / 72.;

        for (auto&& v : c) v /= (h * h);
        for (auto&& v : x) v /= h;

        if (right) {
            ranges::reverse(c);
            ranges::reverse(x);
            for (auto&& v : x) v *= -1;
        }
    }
};

stencil make_E4_2() { return E4_2{}; }

namespace second
{
stencil E4{E4_2{}};
}
} // namespace ccs::stencils
