#include "stencil.hpp"

#include <range/v3/algorithm/copy.hpp>
#include <range/v3/algorithm/fill.hpp>
#include <range/v3/algorithm/reverse.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/repeat.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/transform.hpp>

#include <cmath>

/// E8u - uniform mesh E8 stencil

namespace ccs::stencils
{
struct E8u_1 {

    static constexpr int P = 4;
    static constexpr int R = 7;
    static constexpr int T = 11;
    static constexpr int X = 0;

    std::array<real, 7> alpha;

    E8u_1() = default;
    E8u_1(std::span<const real> a)
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
        c[0] = 1 / (280 * h);
        c[1] = -4 / (105 * h);
        c[2] = 1 / (5 * h);
        c[3] = -4 / (5 * h);
        c[4] = 0;
        c[5] = -c[3];
        c[6] = -c[2];
        c[7] = -c[1];
        c[8] = -c[0];

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
        c[0] = (140 * alpha[0] - 363) / 140;
        c[1] = 7 - 8 * alpha[0];
        c[2] = (56 * alpha[0] - 21) / 2;
        c[3] = -(168 * alpha[0] - 35) / 3;
        c[4] = (280 * alpha[0] - 35) / 4;
        c[5] = -(280 * alpha[0] - 21) / 5;
        c[6] = (168 * alpha[0] - 7) / 6;
        c[7] = -(56 * alpha[0] - 1) / 7;
        c[8] = alpha[0];
        c[9] = 0;
        c[10] = 0;
        c[11] = (7 * alpha[1] - 1) / 7;
        c[12] = -(160 * alpha[1] + 29) / 20;
        c[13] = 28 * alpha[1] + 3;
        c[14] = -(112 * alpha[1] + 5) / 2;
        c[15] = (210 * alpha[1] + 5) / 3;
        c[16] = -(224 * alpha[1] + 3) / 4;
        c[17] = (140 * alpha[1] + 1) / 5;
        c[18] = -(336 * alpha[1] + 1) / 42;
        c[19] = alpha[1];
        c[20] = 0;
        c[21] = 0;
        c[22] = (42 * alpha[2] + 1) / 42;
        c[23] = -(24 * alpha[2] + 1) / 3;
        c[24] = (1680 * alpha[2] - 47) / 60;
        c[25] = -(168 * alpha[2] - 5) / 3;
        c[26] = (420 * alpha[2] - 5) / 6;
        c[27] = -(168 * alpha[2] - 1) / 3;
        c[28] = (336 * alpha[2] - 1) / 12;
        c[29] = -(840 * alpha[2] - 1) / 105;
        c[30] = alpha[2];
        c[31] = 0;
        c[32] = 0;
        c[33] = (105 * alpha[3] - 1) / 105;
        c[34] = -(80 * alpha[3] - 1) / 10;
        c[35] = (140 * alpha[3] - 3) / 5;
        c[36] = -(224 * alpha[3] + 1) / 4;
        c[37] = 70 * alpha[3] + 1;
        c[38] = -(560 * alpha[3] + 3) / 10;
        c[39] = (420 * alpha[3] + 1) / 15;
        c[40] = -(1120 * alpha[3] + 1) / 140;
        c[41] = alpha[3];
        c[42] = 0;
        c[43] = 0;
        c[44] = (140 * alpha[4] + 1) / 140;
        c[45] = -(120 * alpha[4] + 1) / 15;
        c[46] = (280 * alpha[4] + 3) / 10;
        c[47] = (-56 * alpha[4]) - 1;
        c[48] = (280 * alpha[4] + 1) / 4;
        c[49] = -(280 * alpha[4] - 3) / 5;
        c[50] = (280 * alpha[4] - 1) / 10;
        c[51] = -(840 * alpha[4] - 1) / 105;
        c[52] = alpha[4];
        c[53] = 0;
        c[54] = 0;
        c[55] = (840 * alpha[6] + 105 * alpha[5] - 1) / 105;
        c[56] = -(756 * alpha[6] + 96 * alpha[5] - 1) / 12;
        c[57] = (648 * alpha[6] + 84 * alpha[5] - 1) / 3;
        c[58] = -(2520 * alpha[6] + 336 * alpha[5] - 5) / 6;
        c[59] = (1512 * alpha[6] + 210 * alpha[5] - 5) / 3;
        c[60] = -(22680 * alpha[6] + 3360 * alpha[5] - 47) / 60;
        c[61] = (504 * alpha[6] + 84 * alpha[5] + 1) / 3;
        c[62] = -(1512 * alpha[6] + 336 * alpha[5] + 1) / 42;
        c[63] = alpha[5];
        c[64] = alpha[6];
        c[65] = 0;
        c[66] = -(43994496 * alpha[6] + 5499312 * alpha[5] + 3756354 * alpha[4] +
                  7475328 * alpha[3] + 2303742 * alpha[2] + 7419216 * alpha[1] +
                  1545558 * alpha[0] - 28865) /
                5022570;
        c[67] = (8248968 * alpha[6] + 1047488 * alpha[5] + 715496 * alpha[4] +
                 1423872 * alpha[3] + 438808 * alpha[2] + 1413184 * alpha[1] +
                 294392 * alpha[0] - 5917) /
                119585;
        c[68] = -(113128704 * alpha[6] + 14664832 * alpha[5] + 10016944 * alpha[4] +
                  19934208 * alpha[3] + 6143312 * alpha[2] + 19784576 * alpha[1] +
                  4121488 * alpha[0] - 92067) /
                478340;
        c[69] = (164979360 * alpha[6] + 21997248 * alpha[5] + 15025416 * alpha[4] +
                 29901312 * alpha[3] + 9214968 * alpha[2] + 29676864 * alpha[1] +
                 6182232 * alpha[0] - 164197) /
                358755;
        c[70] = -(131983488 * alpha[6] + 18331040 * alpha[5] + 12521180 * alpha[4] +
                  24917760 * alpha[3] + 7679140 * alpha[2] + 24730720 * alpha[1] +
                  5151860 * alpha[0] - 190693) /
                239170;
        c[71] = (49493808 * alpha[6] + 7332416 * alpha[5] + 5008472 * alpha[4] +
                 9967104 * alpha[3] + 3071656 * alpha[2] + 9892288 * alpha[1] +
                 2060744 * alpha[0] - 163203) /
                119585;
        c[72] = -(87988992 * alpha[6] + 14664832 * alpha[5] + 10016944 * alpha[4] +
                  19934208 * alpha[3] + 6143312 * alpha[2] + 19784576 * alpha[1] +
                  4121488 * alpha[0] - 169433) /
                478340;
        c[73] = (32995872 * alpha[6] + 7332416 * alpha[5] + 5008472 * alpha[4] +
                 9967104 * alpha[3] + 3071656 * alpha[2] + 9892288 * alpha[1] +
                 2060744 * alpha[0] + 551009) /
                837095;
        c[74] = -(130936 * alpha[5] + 89437 * alpha[4] + 177984 * alpha[3] +
                  54851 * alpha[2] + 176648 * alpha[1] + 36799 * alpha[0] + 20016) /
                119585;
        c[75] = -(130936 * alpha[6] - 4176) / 119585;
        c[76] = -432.0 / 119585;

        for (auto&& v : c) v /= h;
        if (right) {
            for (auto&& v : c) v *= -1;
            ranges::reverse(c);
        }

        return c;
    }

    std::span<const real> nbs_dirichlet(real h, real, std::span<real> c, bool right) const
    {
        c[0] = (7 * alpha[1] - 1) / 7;
        c[1] = -(160 * alpha[1] + 29) / 20;
        c[2] = 28 * alpha[1] + 3;
        c[3] = -(112 * alpha[1] + 5) / 2;
        c[4] = (210 * alpha[1] + 5) / 3;
        c[5] = -(224 * alpha[1] + 3) / 4;
        c[6] = (140 * alpha[1] + 1) / 5;
        c[7] = -(336 * alpha[1] + 1) / 42;
        c[8] = alpha[1];
        c[9] = 0;
        c[10] = 0;
        c[11] = (42 * alpha[2] + 1) / 42;
        c[12] = -(24 * alpha[2] + 1) / 3;
        c[13] = (1680 * alpha[2] - 47) / 60;
        c[14] = -(168 * alpha[2] - 5) / 3;
        c[15] = (420 * alpha[2] - 5) / 6;
        c[16] = -(168 * alpha[2] - 1) / 3;
        c[17] = (336 * alpha[2] - 1) / 12;
        c[18] = -(840 * alpha[2] - 1) / 105;
        c[19] = alpha[2];
        c[20] = 0;
        c[21] = 0;
        c[22] = (105 * alpha[3] - 1) / 105;
        c[23] = -(80 * alpha[3] - 1) / 10;
        c[24] = (140 * alpha[3] - 3) / 5;
        c[25] = -(224 * alpha[3] + 1) / 4;
        c[26] = 70 * alpha[3] + 1;
        c[27] = -(560 * alpha[3] + 3) / 10;
        c[28] = (420 * alpha[3] + 1) / 15;
        c[29] = -(1120 * alpha[3] + 1) / 140;
        c[30] = alpha[3];
        c[31] = 0;
        c[32] = 0;
        c[33] = (140 * alpha[4] + 1) / 140;
        c[34] = -(120 * alpha[4] + 1) / 15;
        c[35] = (280 * alpha[4] + 3) / 10;
        c[36] = (-56 * alpha[4]) - 1;
        c[37] = (280 * alpha[4] + 1) / 4;
        c[38] = -(280 * alpha[4] - 3) / 5;
        c[39] = (280 * alpha[4] - 1) / 10;
        c[40] = -(840 * alpha[4] - 1) / 105;
        c[41] = alpha[4];
        c[42] = 0;
        c[43] = 0;
        c[44] = (840 * alpha[6] + 105 * alpha[5] - 1) / 105;
        c[45] = -(756 * alpha[6] + 96 * alpha[5] - 1) / 12;
        c[46] = (648 * alpha[6] + 84 * alpha[5] - 1) / 3;
        c[47] = -(2520 * alpha[6] + 336 * alpha[5] - 5) / 6;
        c[48] = (1512 * alpha[6] + 210 * alpha[5] - 5) / 3;
        c[49] = -(22680 * alpha[6] + 3360 * alpha[5] - 47) / 60;
        c[50] = (504 * alpha[6] + 84 * alpha[5] + 1) / 3;
        c[51] = -(1512 * alpha[6] + 336 * alpha[5] + 1) / 42;
        c[52] = alpha[5];
        c[53] = alpha[6];
        c[54] = 0;
        c[55] = -(43994496 * alpha[6] + 5499312 * alpha[5] + 3756354 * alpha[4] +
                  7475328 * alpha[3] + 2303742 * alpha[2] + 7419216 * alpha[1] +
                  1545558 * alpha[0] - 28865) /
                5022570;
        c[56] = (8248968 * alpha[6] + 1047488 * alpha[5] + 715496 * alpha[4] +
                 1423872 * alpha[3] + 438808 * alpha[2] + 1413184 * alpha[1] +
                 294392 * alpha[0] - 5917) /
                119585;
        c[57] = -(113128704 * alpha[6] + 14664832 * alpha[5] + 10016944 * alpha[4] +
                  19934208 * alpha[3] + 6143312 * alpha[2] + 19784576 * alpha[1] +
                  4121488 * alpha[0] - 92067) /
                478340;
        c[58] = (164979360 * alpha[6] + 21997248 * alpha[5] + 15025416 * alpha[4] +
                 29901312 * alpha[3] + 9214968 * alpha[2] + 29676864 * alpha[1] +
                 6182232 * alpha[0] - 164197) /
                358755;
        c[59] = -(131983488 * alpha[6] + 18331040 * alpha[5] + 12521180 * alpha[4] +
                  24917760 * alpha[3] + 7679140 * alpha[2] + 24730720 * alpha[1] +
                  5151860 * alpha[0] - 190693) /
                239170;
        c[60] = (49493808 * alpha[6] + 7332416 * alpha[5] + 5008472 * alpha[4] +
                 9967104 * alpha[3] + 3071656 * alpha[2] + 9892288 * alpha[1] +
                 2060744 * alpha[0] - 163203) /
                119585;
        c[61] = -(87988992 * alpha[6] + 14664832 * alpha[5] + 10016944 * alpha[4] +
                  19934208 * alpha[3] + 6143312 * alpha[2] + 19784576 * alpha[1] +
                  4121488 * alpha[0] - 169433) /
                478340;
        c[62] = (32995872 * alpha[6] + 7332416 * alpha[5] + 5008472 * alpha[4] +
                 9967104 * alpha[3] + 3071656 * alpha[2] + 9892288 * alpha[1] +
                 2060744 * alpha[0] + 551009) /
                837095;
        c[63] = -(130936 * alpha[5] + 89437 * alpha[4] + 177984 * alpha[3] +
                  54851 * alpha[2] + 176648 * alpha[1] + 36799 * alpha[0] + 20016) /
                119585;
        c[64] = -(130936 * alpha[6] - 4176) / 119585;
        c[65] = -432.0 / 119585;
        for (auto&& v : c) v /= h;

        if (right) {
            for (auto&& v : c) v *= -1;
            ranges::reverse(c);
        }

        return c;
    }

    void nbs_neumann(real, real, std::span<real>, std::span<real>, bool) const {}
};

stencil make_E8u_1(std::span<const real> alpha) { return E8u_1{alpha}; }

} // namespace ccs::stencils
