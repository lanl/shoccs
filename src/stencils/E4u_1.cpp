#include "stencil.hpp"

#include <range/v3/algorithm/copy.hpp>
#include <range/v3/algorithm/fill.hpp>
#include <range/v3/algorithm/reverse.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/repeat.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/transform.hpp>

#include <cmath>

/// E4u - uniform mesh E4 stencil

namespace ccs::stencils
{
struct E4u_1 {

    static constexpr int P = 2;
    static constexpr int R = 3;
    static constexpr int T = 5;
    static constexpr int X = 0;

    std::array<real, 2> alpha;

    E4u_1() = default;
    E4u_1(std::span<const real> a)
    {
        rs::copy(vs::concat(a, vs::repeat(0.0)) | vs::take(alpha.size()),
                 rs::begin(alpha));
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
                              std::span<real> x) const
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

    std::span<const real> nbs_floating(real h, real, std::span<real> c, bool right) const
    {

        c[0] = (6 * alpha[0] - 11) / 6;
        c[1] = 3 - 4 * alpha[0];
        c[2] = (12 * alpha[0] - 3) / 2;
        c[3] = -(12 * alpha[0] - 1) / 3;
        c[4] = alpha[0];
        c[5] = (3 * alpha[1] - 1) / 3;
        c[6] = -(8 * alpha[1] + 1) / 2;
        c[7] = 6 * alpha[1] + 1;
        c[8] = -(24 * alpha[1] + 1) / 6;
        c[9] = alpha[1];
        c[10] = -(168 * alpha[1] + 54 * alpha[0] - 11) / 138;
        c[11] = (112 * alpha[1] + 36 * alpha[0] - 15) / 23;
        c[12] = -(336 * alpha[1] + 108 * alpha[0] + 1) / 46;
        c[13] = (336 * alpha[1] + 108 * alpha[0] + 47) / 69;
        c[14] = -(28 * alpha[1] + 9 * alpha[0] + 2) / 23;

        for (auto&& v : c) v /= h;
        if (right) {
            for (auto&& v : c) v *= -1;
            ranges::reverse(c);
        }

        return c;
    }

    std::span<const real> nbs_dirichlet(real h, real, std::span<real> c, bool right) const
    {
        c[0] = (3 * alpha[1] - 1) / 3;
        c[1] = -(8 * alpha[1] + 1) / 2;
        c[2] = 6 * alpha[1] + 1;
        c[3] = -(24 * alpha[1] + 1) / 6;
        c[4] = alpha[1];
        c[5] = -(168 * alpha[1] + 54 * alpha[0] - 11) / 138;
        c[6] = (112 * alpha[1] + 36 * alpha[0] - 15) / 23;
        c[7] = -(336 * alpha[1] + 108 * alpha[0] + 1) / 46;
        c[8] = (336 * alpha[1] + 108 * alpha[0] + 47) / 69;
        c[9] = -(28 * alpha[1] + 9 * alpha[0] + 2) / 23;

        for (auto&& v : c) v /= h;

        if (right) {
            for (auto&& v : c) v *= -1;
            ranges::reverse(c);
        }

        return c;
    }

    void nbs_neumann(real, real, std::span<real>, std::span<real>, bool) const {}
};

stencil make_E4u_1(std::span<const real> alpha) { return E4u_1{alpha}; }

} // namespace ccs::stencils
