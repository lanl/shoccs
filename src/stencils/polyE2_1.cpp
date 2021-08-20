#include "stencil.hpp"

#include <range/v3/algorithm/copy.hpp>
#include <range/v3/algorithm/fill.hpp>
#include <range/v3/algorithm/reverse.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/repeat.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/transform.hpp>

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

    polyE2_1() = default;
    polyE2_1(std::span<const real> fa_, std::span<const real> da_)
    {
        rs::copy(vs::concat(fa_, vs::repeat(0.0)) | vs::take(fa.size()), rs::begin(fa));
        rs::copy(vs::concat(da_, vs::repeat(0.0)) | vs::take(da.size()), rs::begin(da));
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
            // do nothing
            break;
        }
    }

    std::span<const real>
    nbs_floating(real h, real psi, std::span<real> c, bool right) const
    {
        c = c.subspan(0, R * T);

        real t10 = fa[0];
        real t5 = 1 + psi;
        real t6 = 1.0 / (t5);
        real t7 = fa[1];
        real t8 = -1 + t7;
        real t22 = fa[2];
        real t14 = 2 + psi;
        real t19 = fa[3];
        real t20 = -1 + t19;
        real t33 = fa[4];
        real t30 = fa[5];
        real t31 = -1 + t30;
        c[0] = 0.5 * t6 * t8;
        c[1] = (-1 + t10) * 0.5;
        c[2] = -1 * t10 + -0.5 * t14 * t6 * t8;
        c[3] = (t10 + t7) * 0.5;
        c[4] = 0.5 * t20 * t6;
        c[5] = (-1 + t22) * 0.5;
        c[6] = -1 * t22 + -0.5 * t14 * t20 * t6;
        c[7] = (t19 + t22) * 0.5;
        c[8] = 0.5 * t31 * t6;
        c[9] = (-1 + t33) * 0.5;
        c[10] = -1 * t33 + -0.5 * t14 * t31 * t6;
        c[11] = (t30 + t33) * 0.5;

        for (auto&& v : c) v /= h;
        if (right) {
            for (auto&& v : c) v *= -1;
            ranges::reverse(c);
        }
        return c;
    }

    std::span<const real>
    nbs_dirichlet(real h, real psi, std::span<real> c, bool right) const
    {
        c = c.subspan(0, (R - 1) * T);

        real t7 = da[0];
        real t13 = da[1];
        real t19 = da[2];
        real t5 = 1 + psi;
        real t6 = 1.0 / (t5);
        real t8 = -1 + t7;
        real t18 = 2 * psi;
        real t20 = 3 * t19;
        real t21 = 2 * psi * t19;
        real t22 = 1 + t18 + t20 + t21;
        real t23 = 1.0 / (t22);
        real t14 = 2 * psi * t13;
        real t15 = 3 * t13 * t7;
        real t16 = 2 * psi * t13 * t7;
        real t29 = -1 * t13;
        real t25 = 2 + psi;
        real t43 = -1 + t19;
        c[0] = 0.5 * t6 * t8;
        c[1] =
            (-1 + -2 * psi + t13 + t14 + t15 + t16 + -3 * t7 + -2 * psi * t7) * 0.5 * t23;
        c[2] = (-2 * psi * t13 + -3 * t19 + -2 * psi * t19 + t29 + 3 * t7 + 2 * psi * t7 +
                -3 * t13 * t7 + -2 * psi * t13 * t7) *
                   t23 +
             -0.5 * t25 * t6 * t8;
        c[3] = (t13 + t14 + t15 + t16 + t20 + t21 + -2 * t7 + 3 * t19 * t7 +
                2 * psi * t19 * t7) *
               0.5 * t23;
        c[4] = 0.5 * t43 * t6;
        c[5] = (-1 + t13) * 0.5;
        c[6] = t29 + -0.5 * t25 * t43 * t6;
        c[7] = (t13 + t19) * 0.5;

        for (auto&& v : c) v /= h;
        if (right) {
            for (auto&& v : c) v *= -1;
            ranges::reverse(c);
        }
        return c;
    }

    std::span<const real>
    nbs_neumann(real, real, std::span<real> c, std::span<real>, bool) const
    {
        return c;
    }
};

stencil make_polyE2_1(std::span<const real> fa, std::span<const real> da)
{
    return polyE2_1{fa, da};
}

} // namespace ccs::stencils
