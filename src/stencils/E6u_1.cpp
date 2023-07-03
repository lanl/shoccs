#include "stencil.hpp"

#include <range/v3/algorithm/copy.hpp>
#include <range/v3/algorithm/fill.hpp>
#include <range/v3/algorithm/reverse.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/repeat.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/transform.hpp>

#include <cmath>

/// E6u - uniform mesh E6 stencil

namespace ccs::stencils
{
struct E6u_1 {

    static constexpr int P = 3;
    static constexpr int R = 5;
    static constexpr int T = 8;
    static constexpr int X = 0;

    std::array<real, 5> alpha;

    E6u_1() = default;
    E6u_1(std::span<const real> a)
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
        c[0] = -1 / (60 * h);
        c[1] = 3 / (20 * h);
        c[2] = -3 / (4 * h);
        c[3] = 0;
        c[4] = -c[2];
        c[5] = -c[1];
        c[6] = -c[0];

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
        c[0] = (60 * alpha[0] - 137) / 60;
        c[1] = 5 - 6 * alpha[0];
        c[2] = 15 * alpha[0] - 5;
        c[3] = -(60 * alpha[0] - 10) / 3;
        c[4] = (60 * alpha[0] - 5) / 4;
        c[5] = -(30 * alpha[0] - 1) / 5;
        c[6] = alpha[0];
        c[7] = 0;
        c[8] = (5 * alpha[1] - 1) / 5;
        c[9] = -(72 * alpha[1] + 13) / 12;
        c[10] = 15 * alpha[1] + 2;
        c[11] = (-20 * alpha[1]) - 1;
        c[12] = (45 * alpha[1] + 1) / 3;
        c[13] = -(120 * alpha[1] + 1) / 20;
        c[14] = alpha[1];
        c[15] = 0;
        c[16] = (20 * alpha[2] + 1) / 20;
        c[17] = -(12 * alpha[2] + 1) / 2;
        c[18] = (45 * alpha[2] - 1) / 3;
        c[19] = 1 - 20 * alpha[2];
        c[20] = (60 * alpha[2] - 1) / 4;
        c[21] = -(180 * alpha[2] - 1) / 30;
        c[22] = alpha[2];
        c[23] = 0;
        c[24] = (180 * alpha[4] + 30 * alpha[3] - 1) / 30;
        c[25] = -(140 * alpha[4] + 24 * alpha[3] - 1) / 4;
        c[26] = 84 * alpha[4] + 15 * alpha[3] - 1;
        c[27] = -(315 * alpha[4] + 60 * alpha[3] - 1) / 3;
        c[28] = (140 * alpha[4] + 30 * alpha[3] + 1) / 2;
        c[29] = -(420 * alpha[4] + 120 * alpha[3] + 1) / 20;
        c[30] = alpha[3];
        c[31] = alpha[4];
        c[32] = -(190320 * alpha[4] + 31720 * alpha[3] + 22080 * alpha[2] +
                  38040 * alpha[1] + 9500 * alpha[0] - 453) /
                28260;
        c[33] = (55510 * alpha[4] + 9516 * alpha[3] + 6624 * alpha[2] +
                     11412 * alpha[1] + 2850 * alpha[0] - 159) /
                    1413;
        c[34] = -(44408 * alpha[4] + 7930 * alpha[3] + 5520 * alpha[2] + 9510 * alpha[1] +
                  2375 * alpha[0] - 183) /
                471;
        c[35] = (166530 * alpha[4] + 31720 * alpha[3] + 22080 * alpha[2] +
                     38040 * alpha[1] + 9500 * alpha[0] - 1506) /
                    1413;
        c[36] = -(444080 * alpha[4] + 95160 * alpha[3] + 66240 * alpha[2] +
                      114120 * alpha[1] + 28500 * alpha[0] - 1323) /
                    5652;
        c[37] = (55510 * alpha[4] + 15860 * alpha[3] + 11040 * alpha[2] +
                 19020 * alpha[1] + 4750 * alpha[0] + 1551) /
                2355;
        c[38] = -(1586 * alpha[3] + 1104 * alpha[2] + 1902 * alpha[1] +
                      475 * alpha[0] + 192) /
                    1413;
        c[39] = -(1586 * alpha[4] - 24) / 1413;

        for (auto&& v : c) v /= h;
        if (right) {
            for (auto&& v : c) v *= -1;
            ranges::reverse(c);
        }

        return c;
    }

    std::span<const real> nbs_dirichlet(real h, real, std::span<real> c, bool right) const
    {
        // c[0] = (60 * alpha[0] - 137) / 60;
        // c[1] = 5 - 6 * alpha[0];
        // c[2] = 15 * alpha[0] - 5;
        // c[3] = -(60 * alpha[0] - 10) / 3;
        // c[4] = (60 * alpha[0] - 5) / 4;
        // c[5] = -(30 * alpha[0] - 1) / 5;
        // c[6] = alpha[0];
        // c[7] = 0;
        c[0] = (5 * alpha[1] - 1) / 5;
        c[1] = -(72 * alpha[1] + 13) / 12;
        c[2] = 15 * alpha[1] + 2;
        c[3] = (-20 * alpha[1]) - 1;
        c[4] = (45 * alpha[1] + 1) / 3;
        c[5] = -(120 * alpha[1] + 1) / 20;
        c[6] = alpha[1];
        c[7] = 0;
        c[8] = (20 * alpha[2] + 1) / 20;
        c[9] = -(12 * alpha[2] + 1) / 2;
        c[10] = (45 * alpha[2] - 1) / 3;
        c[11] = 1 - 20 * alpha[2];
        c[12] = (60 * alpha[2] - 1) / 4;
        c[13] = -(180 * alpha[2] - 1) / 30;
        c[14] = alpha[2];
        c[15] = 0;
        c[16] = (180 * alpha[4] + 30 * alpha[3] - 1) / 30;
        c[17] = -(140 * alpha[4] + 24 * alpha[3] - 1) / 4;
        c[18] = 84 * alpha[4] + 15 * alpha[3] - 1;
        c[19] = -(315 * alpha[4] + 60 * alpha[3] - 1) / 3;
        c[20] = (140 * alpha[4] + 30 * alpha[3] + 1) / 2;
        c[21] = -(420 * alpha[4] + 120 * alpha[3] + 1) / 20;
        c[22] = alpha[3];
        c[23] = alpha[4];
        c[24] = -(190320 * alpha[4] + 31720 * alpha[3] + 22080 * alpha[2] +
                  38040 * alpha[1] + 9500 * alpha[0] - 453) /
                28260;
        c[25] = (55510 * alpha[4] + 9516 * alpha[3] + 6624 * alpha[2] +
                     11412 * alpha[1] + 2850 * alpha[0] - 159) /
                    1413;
        c[26] = -(44408 * alpha[4] + 7930 * alpha[3] + 5520 * alpha[2] + 9510 * alpha[1] +
                  2375 * alpha[0] - 183) /
                471;
        c[27] = (166530 * alpha[4] + 31720 * alpha[3] + 22080 * alpha[2] +
                     38040 * alpha[1] + 9500 * alpha[0] - 1506) /
                    1413;
        c[28] = -(444080 * alpha[4] + 95160 * alpha[3] + 66240 * alpha[2] +
                      114120 * alpha[1] + 28500 * alpha[0] - 1323) /
                    5652;
        c[29] = (55510 * alpha[4] + 15860 * alpha[3] + 11040 * alpha[2] +
                 19020 * alpha[1] + 4750 * alpha[0] + 1551) /
                2355;
        c[30] = -(1586 * alpha[3] + 1104 * alpha[2] + 1902 * alpha[1] +
                      475 * alpha[0] + 192) /
                    1413;
        c[31] = -(1586 * alpha[4] - 24) / 1413;

        for (auto&& v : c) v /= h;

        if (right) {
            for (auto&& v : c) v *= -1;
            ranges::reverse(c);
        }

        return c;
    }

    void nbs_neumann(real, real, std::span<real>, std::span<real>, bool) const {}
};

stencil make_E6u_1(std::span<const real> alpha) { return E6u_1{alpha}; }

} // namespace ccs::stencils
