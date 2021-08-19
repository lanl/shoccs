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
struct E2_1 {

    static constexpr int P = 1;
    static constexpr int R = 4;
    static constexpr int T = 5;
    static constexpr int X = 0;

    std::array<real, 4> alpha;

    E2_1() = default;
    E2_1(std::span<const real> a)
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
    interp_info query_interp() const { return {2, 3}; }

    std::span<const real> interp_interior(real y, std::span<real> c) const
    {
        if (y > 0) {
            c[0] = 1 + -1 * y;
            c[1] = y;
        } else {
            c[0] = -1 * y;
            c[1] = 1 + y;
        }
        return c.subspan(0, 2);
    }

    std::span<const real>
    interp_wall(int i, real y, real psi, std::span<real> c, bool right) const
    {
        if (right) {
            const real t6 = 1 + psi;
            const real t7 = 1.0 / (t6);
            const real t8 = -1 + y;
            const real t9 = psi * t8;
            const real t5 = -1 + psi;
            const real t12 = -1 * psi * y;
            const real t20 = 1 + y;
            switch (i) {
            case 0:
                c[0] = (psi + y + psi * y) * t5;
                c[1] = 1 + t12 + -1 * psi * psi * t20 + y;
                c[2] = psi * t20;
                break;
            case 1:
                c[0] = (t9 + y) * t5 * t7;
                c[1] = psi + t12;
                c[2] = (1 + t9 + y) * t7;
                break;
            }
        } else {
            const real t8 = -1 + y;
            const real t11 = psi * y;
            const real t13 = -1 + psi;
            const real t17 = 1 + psi;
            const real t18 = 1.0 / (t17);
            switch (i) {
            case 0:
                c[0] = psi + -1 * psi * y;
                c[1] = 1 + t11 + psi * psi * t8 + -1 * y;
                c[2] = (psi * t8 + y) * -1 * t13;
                break;
            case 1:
                c[0] = (-1 + psi + t11 + y) * -1 * t18;
                c[1] = (1 + y) * psi;
                c[2] = (psi + t11 + y) * -1 * t13 * t18;
                break;
            }
        }
        return c.subspan(0, 3);
    }

    std::span<const real> interior(real h, std::span<real> c) const
    {
        c[0] = -1 / (2 * h);
        c[1] = 0;
        c[2] = 1 / (2 * h);

        return c;
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
            return c
            // do nothing
            break;
        }
    }

    std::span<const real> nbs_floating(real h, real psi, std::span<real> c, bool right) const
    {
        double t3 = alpha[0];
        double t5 = alpha[2];
        double t17 = -1 + psi;
        double t11 = -psi;
        double t22 = alpha[1];
        double t9 = 2 * t5;
        double t24 = alpha[3];
        double t28 = 1 + psi;
        double t29 = std::pow(t28, -1);
        double t12 = -2 * t3;
        double t36 = std::pow(psi, 2);
        double t14 = -3 * t5;
        double t18 = -(t17 * t3);
        double t21 = -(t17 * t5);
        double t53 = -6 * t3;
        double t54 = -3 * t22;
        double t55 = 5 * t22 * t3;
        double t56 = -14 * t5;
        double t57 = 10 * t22 * t5;
        double t58 = -9 * t24;
        double t59 = 15 * t24 * t3;
        double t60 = 30 * t24 * t5;
        double t61 = 4 + t53 + t54 + t55 + t56 + t57 + t58 + t59 + t60;
        double t62 = std::pow(t61, -1);
        double t37 = -t36;
        double t73 = std::pow(t22, 2);
        double t44 = 2 * t24 * t36;
        double t100 = std::pow(t24, 2);
        double t49 = 3 * t3;
        double t50 = -1 + t49 + t9;
        double t51 = 2 * t24;
        double t52 = -1 + t22 + t51;
        double t13 = 3 * psi * t3;
        double t153 = std::pow(psi, 3);
        double t161 = std::pow(t3, 2);
        double t159 = std::pow(psi, 4);
        double t167 = std::pow(t3, 3);
        double t208 = std::pow(t22, 3);
        double t266 = std::pow(t5, 2);
        double t298 = std::pow(t5, 3);
        double t491 = std::pow(t24, 3);
        double t222 = -6 * t159 * t5;
        double t237 = 6 * t159 * t22 * t5;
        double t292 = -960 * psi * t266 * t3 * t73;
        double t394 = -1920 * t24 * t3 * t36 * t5 * t73;
        double t483 = 2160 * t100 * t153 * t266 * t3;
        double t494 = -432 * t153 * t491;
        double t508 = -4320 * t266 * t36 * t491;
        double t151 = -27 * psi;
        double t152 = 11 * t36;
        double t154 = 6 * t153;
        double t155 = 128 * t3;
        double t156 = 36 * psi * t3;
        double t157 = -31 * t3 * t36;
        double t158 = -11 * t153 * t3;
        double t160 = -2 * t159 * t3;
        double t162 = -168 * t161;
        double t163 = 71 * psi * t161;
        double t164 = -3 * t161 * t36;
        double t165 = 14 * t153 * t161;
        double t166 = 8 * t159 * t161;
        double t168 = 72 * t167;
        double t169 = -96 * psi * t167;
        double t170 = 39 * t167 * t36;
        double t171 = -9 * t153 * t167;
        double t172 = -6 * t159 * t167;
        double t173 = 48 * t22;
        double t174 = 102 * psi * t22;
        double t175 = -48 * t22 * t36;
        double t176 = -32 * t153 * t22;
        double t177 = -200 * t22 * t3;
        double t178 = -239 * psi * t22 * t3;
        double t179 = 180 * t22 * t3 * t36;
        double t180 = 57 * t153 * t22 * t3;
        double t181 = 2 * t159 * t22 * t3;
        double t182 = 272 * t161 * t22;
        double t183 = 14 * psi * t161 * t22;
        double t184 = -136 * t161 * t22 * t36;
        double t185 = -12 * t153 * t161 * t22;
        double t186 = -8 * t159 * t161 * t22;
        double t187 = -120 * t167 * t22;
        double t188 = 171 * psi * t167 * t22;
        double t189 = -44 * t167 * t22 * t36;
        double t190 = -13 * t153 * t167 * t22;
        double t191 = 6 * t159 * t167 * t22;
        double t192 = -18 * t73;
        double t193 = -108 * psi * t73;
        double t194 = 46 * t36 * t73;
        double t195 = 52 * t153 * t73;
        double t196 = 78 * t3 * t73;
        double t197 = 312 * psi * t3 * t73;
        double t198 = -194 * t3 * t36 * t73;
        double t199 = -116 * t153 * t3 * t73;
        double t200 = -110 * t161 * t73;
        double t201 = -172 * psi * t161 * t73;
        double t202 = 186 * t161 * t36 * t73;
        double t203 = 44 * t153 * t161 * t73;
        double t204 = 50 * t167 * t73;
        double t205 = -80 * psi * t167 * t73;
        double t206 = 10 * t167 * t36 * t73;
        double t207 = 20 * t153 * t167 * t73;
        double t209 = 36 * psi * t208;
        double t210 = -12 * t208 * t36;
        double t211 = -24 * t153 * t208;
        double t212 = -120 * psi * t208 * t3;
        double t213 = 56 * t208 * t3 * t36;
        double t214 = 64 * t153 * t208 * t3;
        double t215 = 100 * psi * t161 * t208;
        double t216 = -60 * t161 * t208 * t36;
        double t217 = -40 * t153 * t161 * t208;
        double t218 = 288 * t5;
        double t219 = 75 * psi * t5;
        double t220 = -104 * t36 * t5;
        double t221 = -37 * t153 * t5;
        double t223 = -752 * t3 * t5;
        double t224 = 340 * psi * t3 * t5;
        double t225 = 42 * t3 * t36 * t5;
        double t226 = 50 * t153 * t3 * t5;
        double t227 = 32 * t159 * t3 * t5;
        double t228 = 480 * t161 * t5;
        double t229 = -667 * psi * t161 * t5;
        double t230 = 258 * t161 * t36 * t5;
        double t231 = -37 * t153 * t161 * t5;
        double t232 = -34 * t159 * t161 * t5;
        double t233 = -424 * t22 * t5;
        double t234 = -539 * psi * t22 * t5;
        double t235 = 440 * t22 * t36 * t5;
        double t236 = 157 * t153 * t22 * t5;
        double t238 = 1152 * t22 * t3 * t5;
        double t239 = 104 * psi * t22 * t3 * t5;
        double t240 = -678 * t22 * t3 * t36 * t5;
        double t241 = -66 * t153 * t22 * t3 * t5;
        double t242 = -32 * t159 * t22 * t3 * t5;
        double t243 = -760 * t161 * t22 * t5;
        double t244 = 1103 * psi * t161 * t22 * t5;
        double t245 = -278 * t161 * t22 * t36 * t5;
        double t246 = -99 * t153 * t161 * t22 * t5;
        double t247 = 34 * t159 * t161 * t22 * t5;
        double t248 = 156 * t5 * t73;
        double t249 = 672 * psi * t5 * t73;
        double t250 = -420 * t36 * t5 * t73;
        double t251 = -264 * t153 * t5 * t73;
        double t252 = -440 * t3 * t5 * t73;
        double t253 = -768 * psi * t3 * t5 * t73;
        double t254 = 808 * t3 * t36 * t5 * t73;
        double t255 = 208 * t153 * t3 * t5 * t73;
        double t256 = 300 * t161 * t5 * t73;
        double t257 = -480 * psi * t161 * t5 * t73;
        double t258 = 60 * t161 * t36 * t5 * t73;
        double t259 = 120 * t153 * t161 * t5 * t73;
        double t260 = -240 * psi * t208 * t5;
        double t261 = 112 * t208 * t36 * t5;
        double t262 = 128 * t153 * t208 * t5;
        double t263 = 400 * psi * t208 * t3 * t5;
        double t264 = -240 * t208 * t3 * t36 * t5;
        double t265 = -160 * t153 * t208 * t3 * t5;
        double t267 = -840 * t266;
        double t268 = 408 * psi * t266;
        double t269 = 112 * t266 * t36;
        double t270 = 32 * t153 * t266;
        double t271 = 24 * t159 * t266;
        double t272 = 1064 * t266 * t3;
        double t273 = -1544 * psi * t266 * t3;
        double t274 = 556 * t266 * t3 * t36;
        double t275 = -20 * t153 * t266 * t3;
        double t276 = -56 * t159 * t266 * t3;
        double t277 = 1216 * t22 * t266;
        double t278 = 148 * psi * t22 * t266;
        double t279 = -828 * t22 * t266 * t36;
        double t280 = -72 * t153 * t22 * t266;
        double t281 = -24 * t159 * t22 * t266;
        double t282 = -1600 * t22 * t266 * t3;
        double t283 = 2380 * psi * t22 * t266 * t3;
        double t284 = -576 * t22 * t266 * t3 * t36;
        double t285 = -260 * t153 * t22 * t266 * t3;
        double t286 = 56 * t159 * t22 * t266 * t3;
        double t287 = -440 * t266 * t73;
        double t288 = -848 * psi * t266 * t73;
        double t289 = 872 * t266 * t36 * t73;
        double t290 = 240 * t153 * t266 * t73;
        double t291 = 600 * t266 * t3 * t73;
        double t293 = 120 * t266 * t3 * t36 * t73;
        double t294 = 240 * t153 * t266 * t3 * t73;
        double t295 = 400 * psi * t208 * t266;
        double t296 = -240 * t208 * t266 * t36;
        double t297 = -160 * t153 * t208 * t266;
        double t299 = 784 * t298;
        double t300 = -1188 * psi * t298;
        double t301 = 392 * t298 * t36;
        double t302 = 36 * t153 * t298;
        double t303 = -24 * t159 * t298;
        double t304 = -1120 * t22 * t298;
        double t305 = 1716 * psi * t22 * t298;
        double t306 = -392 * t22 * t298 * t36;
        double t307 = -228 * t153 * t22 * t298;
        double t308 = 24 * t159 * t22 * t298;
        double t309 = 400 * t298 * t73;
        double t310 = -640 * psi * t298 * t73;
        double t311 = 80 * t298 * t36 * t73;
        double t312 = 160 * t153 * t298 * t73;
        double t313 = 144 * t24;
        double t314 = 235 * psi * t24;
        double t315 = -132 * t24 * t36;
        double t316 = -81 * t153 * t24;
        double t317 = 2 * t159 * t24;
        double t318 = -600 * t24 * t3;
        double t319 = -487 * psi * t24 * t3;
        double t320 = 485 * t24 * t3 * t36;
        double t321 = 126 * t153 * t24 * t3;
        double t322 = -4 * t159 * t24 * t3;
        double t323 = 816 * t161 * t24;
        double t324 = -163 * psi * t161 * t24;
        double t325 = -350 * t161 * t24 * t36;
        double t326 = 19 * t153 * t161 * t24;
        double t327 = -10 * t159 * t161 * t24;
        double t328 = -360 * t167 * t24;
        double t329 = 543 * psi * t167 * t24;
        double t330 = -135 * t167 * t24 * t36;
        double t331 = -60 * t153 * t167 * t24;
        double t332 = 12 * t159 * t167 * t24;
        double t333 = -108 * t22 * t24;
        double t334 = -552 * psi * t22 * t24;
        double t335 = 251 * t22 * t24 * t36;
        double t336 = 271 * t153 * t22 * t24;
        double t337 = -2 * t159 * t22 * t24;
        double t338 = 468 * t22 * t24 * t3;
        double t339 = 1568 * psi * t22 * t24 * t3;
        double t340 = -1050 * t22 * t24 * t3 * t36;
        double t341 = -594 * t153 * t22 * t24 * t3;
        double t342 = 8 * t159 * t22 * t24 * t3;
        double t343 = -660 * t161 * t22 * t24;
        double t344 = -792 * psi * t161 * t22 * t24;
        double t345 = 999 * t161 * t22 * t24 * t36;
        double t346 = 199 * t153 * t161 * t22 * t24;
        double t347 = -6 * t159 * t161 * t22 * t24;
        double t348 = 300 * t167 * t22 * t24;
        double t349 = -480 * psi * t167 * t22 * t24;
        double t350 = 60 * t167 * t22 * t24 * t36;
        double t351 = 120 * t153 * t167 * t22 * t24;
        double t352 = 288 * psi * t24 * t73;
        double t353 = -96 * t24 * t36 * t73;
        double t354 = -192 * t153 * t24 * t73;
        double t355 = -960 * psi * t24 * t3 * t73;
        double t356 = 448 * t24 * t3 * t36 * t73;
        double t357 = 512 * t153 * t24 * t3 * t73;
        double t358 = 800 * psi * t161 * t24 * t73;
        double t359 = -480 * t161 * t24 * t36 * t73;
        double t360 = -320 * t153 * t161 * t24 * t73;
        double t361 = -1272 * t24 * t5;
        double t362 = -1116 * psi * t24 * t5;
        double t363 = 1154 * t24 * t36 * t5;
        double t364 = 366 * t153 * t24 * t5;
        double t365 = 4 * t159 * t24 * t5;
        double t366 = 3456 * t24 * t3 * t5;
        double t367 = -556 * psi * t24 * t3 * t5;
        double t368 = -1712 * t24 * t3 * t36 * t5;
        double t369 = 12 * t153 * t24 * t3 * t5;
        double t370 = -48 * t159 * t24 * t3 * t5;
        double t371 = -2280 * t161 * t24 * t5;
        double t372 = 3464 * psi * t161 * t24 * t5;
        double t373 = -842 * t161 * t24 * t36 * t5;
        double t374 = -410 * t153 * t161 * t24 * t5;
        double t375 = 68 * t159 * t161 * t24 * t5;
        double t376 = 936 * t22 * t24 * t5;
        double t377 = 3376 * psi * t22 * t24 * t5;
        double t378 = -2248 * t22 * t24 * t36 * t5;
        double t379 = -1352 * t153 * t22 * t24 * t5;
        double t380 = 8 * t159 * t22 * t24 * t5;
        double t381 = -2640 * t22 * t24 * t3 * t5;
        double t382 = -3568 * psi * t22 * t24 * t3 * t5;
        double t383 = 4296 * t22 * t24 * t3 * t36 * t5;
        double t384 = 968 * t153 * t22 * t24 * t3 * t5;
        double t385 = -16 * t159 * t22 * t24 * t3 * t5;
        double t386 = 1800 * t161 * t22 * t24 * t5;
        double t387 = -2880 * psi * t161 * t22 * t24 * t5;
        double t388 = 360 * t161 * t22 * t24 * t36 * t5;
        double t389 = 720 * t153 * t161 * t22 * t24 * t5;
        double t390 = -1920 * psi * t24 * t5 * t73;
        double t391 = 896 * t24 * t36 * t5 * t73;
        double t392 = 1024 * t153 * t24 * t5 * t73;
        double t393 = 3200 * psi * t24 * t3 * t5 * t73;
        double t395 = -1280 * t153 * t24 * t3 * t5 * t73;
        double t396 = 3648 * t24 * t266;
        double t397 = -468 * psi * t24 * t266;
        double t398 = -2056 * t24 * t266 * t36;
        double t399 = -28 * t153 * t24 * t266;
        double t400 = -40 * t159 * t24 * t266;
        double t401 = -4800 * t24 * t266 * t3;
        double t402 = 7380 * psi * t24 * t266 * t3;
        double t403 = -1732 * t24 * t266 * t3 * t36;
        double t404 = -960 * t153 * t24 * t266 * t3;
        double t405 = 112 * t159 * t24 * t266 * t3;
        double t406 = -2640 * t22 * t24 * t266;
        double t407 = -3968 * psi * t22 * t24 * t266;
        double t408 = 4596 * t22 * t24 * t266 * t36;
        double t409 = 1140 * t153 * t22 * t24 * t266;
        double t410 = -8 * t159 * t22 * t24 * t266;
        double t411 = 3600 * t22 * t24 * t266 * t3;
        double t412 = -5760 * psi * t22 * t24 * t266 * t3;
        double t413 = 720 * t22 * t24 * t266 * t3 * t36;
        double t414 = 1440 * t153 * t22 * t24 * t266 * t3;
        double t415 = 3200 * psi * t24 * t266 * t73;
        double t416 = -1920 * t24 * t266 * t36 * t73;
        double t417 = -1280 * t153 * t24 * t266 * t73;
        double t418 = -3360 * t24 * t298;
        double t419 = 5248 * psi * t24 * t298;
        double t420 = -1176 * t24 * t298 * t36;
        double t421 = -760 * t153 * t24 * t298;
        double t422 = 48 * t159 * t24 * t298;
        double t423 = 2400 * t22 * t24 * t298;
        double t424 = -3840 * psi * t22 * t24 * t298;
        double t425 = 480 * t22 * t24 * t298 * t36;
        double t426 = 960 * t153 * t22 * t24 * t298;
        double t427 = -162 * t100;
        double t428 = -684 * psi * t100;
        double t429 = 336 * t100 * t36;
        double t430 = 346 * t100 * t153;
        double t431 = -4 * t100 * t159;
        double t432 = 702 * t100 * t3;
        double t433 = 1896 * psi * t100 * t3;
        double t434 = -1390 * t100 * t3 * t36;
        double t435 = -744 * t100 * t153 * t3;
        double t436 = 16 * t100 * t159 * t3;
        double t437 = -990 * t100 * t161;
        double t438 = -828 * psi * t100 * t161;
        double t439 = 1308 * t100 * t161 * t36;
        double t440 = 210 * t100 * t153 * t161;
        double t441 = -12 * t100 * t159 * t161;
        double t442 = 450 * t100 * t167;
        double t443 = -720 * psi * t100 * t167;
        double t444 = 90 * t100 * t167 * t36;
        double t445 = 180 * t100 * t153 * t167;
        double t446 = 756 * psi * t100 * t22;
        double t447 = -252 * t100 * t22 * t36;
        double t448 = -504 * t100 * t153 * t22;
        double t449 = -2520 * psi * t100 * t22 * t3;
        double t450 = 1176 * t100 * t22 * t3 * t36;
        double t451 = 1344 * t100 * t153 * t22 * t3;
        double t452 = 2100 * psi * t100 * t161 * t22;
        double t453 = -1260 * t100 * t161 * t22 * t36;
        double t454 = -840 * t100 * t153 * t161 * t22;
        double t455 = 1404 * t100 * t5;
        double t456 = 4080 * psi * t100 * t5;
        double t457 = -2948 * t100 * t36 * t5;
        double t458 = -1688 * t100 * t153 * t5;
        double t459 = 16 * t100 * t159 * t5;
        double t460 = -3960 * t100 * t3 * t5;
        double t461 = -3792 * psi * t100 * t3 * t5;
        double t462 = 5576 * t100 * t3 * t36 * t5;
        double t463 = 1056 * t100 * t153 * t3 * t5;
        double t464 = -32 * t100 * t159 * t3 * t5;
        double t465 = 2700 * t100 * t161 * t5;
        double t466 = -4320 * psi * t100 * t161 * t5;
        double t467 = 540 * t100 * t161 * t36 * t5;
        double t468 = 1080 * t100 * t153 * t161 * t5;
        double t469 = -5040 * psi * t100 * t22 * t5;
        double t470 = 2352 * t100 * t22 * t36 * t5;
        double t471 = 2688 * t100 * t153 * t22 * t5;
        double t472 = 8400 * psi * t100 * t22 * t3 * t5;
        double t473 = -5040 * t100 * t22 * t3 * t36 * t5;
        double t474 = -3360 * t100 * t153 * t22 * t3 * t5;
        double t475 = -3960 * t100 * t266;
        double t476 = -4272 * psi * t100 * t266;
        double t477 = 5920 * t100 * t266 * t36;
        double t478 = 1272 * t100 * t153 * t266;
        double t479 = -16 * t100 * t159 * t266;
        double t480 = 5400 * t100 * t266 * t3;
        double t481 = -8640 * psi * t100 * t266 * t3;
        double t482 = 1080 * t100 * t266 * t3 * t36;
        double t484 = 8400 * psi * t100 * t22 * t266;
        double t485 = -5040 * t100 * t22 * t266 * t36;
        double t486 = -3360 * t100 * t153 * t22 * t266;
        double t487 = 3600 * t100 * t298;
        double t488 = -5760 * psi * t100 * t298;
        double t489 = 720 * t100 * t298 * t36;
        double t490 = 1440 * t100 * t153 * t298;
        double t492 = 648 * psi * t491;
        double t493 = -216 * t36 * t491;
        double t495 = -2160 * psi * t3 * t491;
        double t496 = 1008 * t3 * t36 * t491;
        double t497 = 1152 * t153 * t3 * t491;
        double t498 = 1800 * psi * t161 * t491;
        double t499 = -1080 * t161 * t36 * t491;
        double t500 = -720 * t153 * t161 * t491;
        double t501 = -4320 * psi * t491 * t5;
        double t502 = 2016 * t36 * t491 * t5;
        double t503 = 2304 * t153 * t491 * t5;
        double t504 = 7200 * psi * t3 * t491 * t5;
        double t505 = -4320 * t3 * t36 * t491 * t5;
        double t506 = -2880 * t153 * t3 * t491 * t5;
        double t507 = 7200 * psi * t266 * t491;
        double t509 = -2880 * t153 * t266 * t491;
        double t510 =
            -32 + t151 + t152 + t154 + t155 + t156 + t157 + t158 + t160 + t162 + t163 +
            t164 + t165 + t166 + t168 + t169 + t170 + t171 + t172 + t173 + t174 + t175 +
            t176 + t177 + t178 + t179 + t180 + t181 + t182 + t183 + t184 + t185 + t186 +
            t187 + t188 + t189 + t190 + t191 + t192 + t193 + t194 + t195 + t196 + t197 +
            t198 + t199 + t200 + t201 + t202 + t203 + t204 + t205 + t206 + t207 + t209 +
            t210 + t211 + t212 + t213 + t214 + t215 + t216 + t217 + t218 + t219 + t220 +
            t221 + t222 + t223 + t224 + t225 + t226 + t227 + t228 + t229 + t230 + t231 +
            t232 + t233 + t234 + t235 + t236 + t237 + t238 + t239 + t240 + t241 + t242 +
            t243 + t244 + t245 + t246 + t247 + t248 + t249 + t250 + t251 + t252 + t253 +
            t254 + t255 + t256 + t257 + t258 + t259 + t260 + t261 + t262 + t263 + t264 +
            t265 + t267 + t268 + t269 + t270 + t271 + t272 + t273 + t274 + t275 + t276 +
            t277 + t278 + t279 + t280 + t281 + t282 + t283 + t284 + t285 + t286 + t287 +
            t288 + t289 + t290 + t291 + t292 + t293 + t294 + t295 + t296 + t297 + t299 +
            t300 + t301 + t302 + t303 + t304 + t305 + t306 + t307 + t308 + t309 + t310 +
            t311 + t312 + t313 + t314 + t315 + t316 + t317 + t318 + t319 + t320 + t321 +
            t322 + t323 + t324 + t325 + t326 + t327 + t328 + t329 + t330 + t331 + t332 +
            t333 + t334 + t335 + t336 + t337 + t338 + t339 + t340 + t341 + t342 + t343 +
            t344 + t345 + t346 + t347 + t348 + t349 + t350 + t351 + t352 + t353 + t354 +
            t355 + t356 + t357 + t358 + t359 + t360 + t361 + t362 + t363 + t364 + t365 +
            t366 + t367 + t368 + t369 + t370 + t371 + t372 + t373 + t374 + t375 + t376 +
            t377 + t378 + t379 + t380 + t381 + t382 + t383 + t384 + t385 + t386 + t387 +
            t388 + t389 + t390 + t391 + t392 + t393 + t394 + t395 + t396 + t397 + t398 +
            t399 + t400 + t401 + t402 + t403 + t404 + t405 + t406 + t407 + t408 + t409 +
            t410 + t411 + t412 + t413 + t414 + t415 + t416 + t417 + t418 + t419 + t420 +
            t421 + t422 + t423 + t424 + t425 + t426 + t427 + t428 + t429 + t430 + t431 +
            t432 + t433 + t434 + t435 + t436 + t437 + t438 + t439 + t440 + t441 + t442 +
            t443 + t444 + t445 + t446 + t447 + t448 + t449 + t450 + t451 + t452 + t453 +
            t454 + t455 + t456 + t457 + t458 + t459 + t460 + t461 + t462 + t463 + t464 +
            t465 + t466 + t467 + t468 + t469 + t470 + t471 + t472 + t473 + t474 + t475 +
            t476 + t477 + t478 + t479 + t480 + t481 + t482 + t483 + t484 + t485 + t486 +
            t487 + t488 + t489 + t490 + t492 + t493 + t494 + t495 + t496 + t497 + t498 +
            t499 + t500 + t501 + t502 + t503 + t504 + t505 + t506 + t507 + t508 + t509;
        double t511 = std::pow(t510, -1);
        double t522 = 36 * t167;
        double t67 = 14 * t22;
        double t74 = -6 * t73;
        double t551 = -40 * psi * t167 * t73;
        double t86 = 34 * t24;
        double t91 = -30 * t22 * t24;
        double t677 = 150 * t167 * t22 * t24;
        double t718 = 360 * t153 * t161 * t22 * t24 * t5;
        double t101 = -36 * t100;
        double t512 = -21 * psi;
        double t513 = 3 * t36;
        double t514 = 84 * t3;
        double t515 = 32 * psi * t3;
        double t516 = 6 * t3 * t36;
        double t517 = -2 * t153 * t3;
        double t518 = -96 * t161;
        double t519 = 25 * psi * t161;
        double t520 = -15 * t161 * t36;
        double t521 = 8 * t153 * t161;
        double t523 = -36 * psi * t167;
        double t524 = 6 * t167 * t36;
        double t525 = -6 * t153 * t167;
        double t526 = 34 * t22;
        double t527 = 65 * psi * t22;
        double t528 = -29 * t22 * t36;
        double t529 = -122 * t22 * t3;
        double t530 = -120 * psi * t22 * t3;
        double t531 = 40 * t22 * t3 * t36;
        double t532 = 2 * t153 * t22 * t3;
        double t533 = 142 * t161 * t22;
        double t534 = -21 * psi * t161 * t22;
        double t535 = 17 * t161 * t22 * t36;
        double t536 = -8 * t153 * t161 * t22;
        double t537 = -54 * t167 * t22;
        double t538 = 76 * psi * t167 * t22;
        double t539 = -28 * t167 * t22 * t36;
        double t540 = 6 * t153 * t167 * t22;
        double t541 = -12 * t73;
        double t542 = -68 * psi * t73;
        double t543 = 52 * t36 * t73;
        double t544 = 44 * t3 * t73;
        double t545 = 152 * psi * t3 * t73;
        double t546 = -116 * t3 * t36 * t73;
        double t547 = -52 * t161 * t73;
        double t548 = -44 * psi * t161 * t73;
        double t549 = 44 * t161 * t36 * t73;
        double t550 = 20 * t167 * t73;
        double t552 = 20 * t167 * t36 * t73;
        double t553 = 24 * psi * t208;
        double t554 = -24 * t208 * t36;
        double t555 = -64 * psi * t208 * t3;
        double t556 = 64 * t208 * t3 * t36;
        double t557 = 40 * psi * t161 * t208;
        double t558 = -40 * t161 * t208 * t36;
        double t559 = 228 * t5;
        double t560 = 49 * psi * t5;
        double t561 = -54 * t36 * t5;
        double t562 = -(t153 * t5);
        double t563 = -528 * t3 * t5;
        double t564 = 228 * psi * t3 * t5;
        double t565 = -72 * t3 * t36 * t5;
        double t566 = 60 * t153 * t3 * t5;
        double t567 = 24 * t159 * t3 * t5;
        double t568 = 300 * t161 * t5;
        double t569 = -377 * psi * t161 * t5;
        double t570 = 162 * t161 * t36 * t5;
        double t571 = -67 * t153 * t161 * t5;
        double t572 = -18 * t159 * t161 * t5;
        double t573 = -320 * t22 * t5;
        double t574 = -309 * psi * t22 * t5;
        double t575 = 314 * t22 * t36 * t5;
        double t576 = -51 * t153 * t22 * t5;
        double t577 = 760 * t22 * t3 * t5;
        double t578 = -164 * psi * t22 * t3 * t5;
        double t579 = -72 * t22 * t3 * t36 * t5;
        double t580 = -20 * t153 * t22 * t3 * t5;
        double t581 = -24 * t159 * t22 * t3 * t5;
        double t582 = -440 * t161 * t22 * t5;
        double t583 = 733 * psi * t161 * t22 * t5;
        double t584 = -422 * t161 * t22 * t36 * t5;
        double t585 = 111 * t153 * t161 * t22 * t5;
        double t586 = 18 * t159 * t161 * t22 * t5;
        double t587 = 112 * t5 * t73;
        double t588 = 432 * psi * t5 * t73;
        double t589 = -504 * t36 * t5 * t73;
        double t590 = 104 * t153 * t5 * t73;
        double t591 = -272 * t3 * t5 * t73;
        double t592 = -296 * psi * t3 * t5 * t73;
        double t593 = 504 * t3 * t36 * t5 * t73;
        double t594 = -128 * t153 * t3 * t5 * t73;
        double t595 = 160 * t161 * t5 * t73;
        double t596 = -360 * psi * t161 * t5 * t73;
        double t597 = 240 * t161 * t36 * t5 * t73;
        double t598 = -40 * t153 * t161 * t5 * t73;
        double t599 = -176 * psi * t208 * t5;
        double t600 = 224 * t208 * t36 * t5;
        double t601 = -48 * t153 * t208 * t5;
        double t602 = 240 * psi * t208 * t3 * t5;
        double t603 = -320 * t208 * t3 * t36 * t5;
        double t604 = 80 * t153 * t208 * t3 * t5;
        double t605 = -696 * t266;
        double t606 = 416 * psi * t266;
        double t607 = -32 * t266 * t36;
        double t608 = 28 * t153 * t266;
        double t609 = 20 * t159 * t266;
        double t610 = 792 * t266 * t3;
        double t611 = -1184 * psi * t266 * t3;
        double t612 = 612 * t266 * t3 * t36;
        double t613 = -184 * t153 * t266 * t3;
        double t614 = -36 * t159 * t266 * t3;
        double t615 = 968 * t22 * t266;
        double t616 = -264 * psi * t22 * t266;
        double t617 = -352 * t22 * t266 * t36;
        double t618 = 108 * t153 * t22 * t266;
        double t619 = -20 * t159 * t22 * t266;
        double t620 = -1128 * t22 * t266 * t3;
        double t621 = 2120 * psi * t22 * t266 * t3;
        double t622 = -1396 * t22 * t266 * t3 * t36;
        double t623 = 368 * t153 * t22 * t266 * t3;
        double t624 = 36 * t159 * t22 * t266 * t3;
        double t625 = -336 * t266 * t73;
        double t626 = -448 * psi * t266 * t73;
        double t627 = 928 * t266 * t36 * t73;
        double t628 = -320 * t153 * t266 * t73;
        double t629 = 400 * t266 * t3 * t73;
        double t630 = 720 * t266 * t3 * t36 * t73;
        double t631 = -160 * t153 * t266 * t3 * t73;
        double t632 = 320 * psi * t208 * t266;
        double t633 = -480 * t208 * t266 * t36;
        double t634 = 160 * t153 * t208 * t266;
        double t635 = 672 * t298;
        double t636 = -1148 * psi * t298;
        double t637 = 640 * t298 * t36;
        double t638 = -148 * t153 * t298;
        double t639 = -16 * t159 * t298;
        double t640 = -928 * t22 * t298;
        double t641 = 1916 * psi * t22 * t298;
        double t642 = -1344 * t22 * t298 * t36;
        double t643 = 340 * t153 * t22 * t298;
        double t644 = 16 * t159 * t22 * t298;
        double t645 = 320 * t298 * t73;
        double t646 = -800 * psi * t298 * t73;
        double t647 = 640 * t298 * t36 * t73;
        double t648 = -160 * t153 * t298 * t73;
        double t649 = 110 * t24;
        double t650 = 190 * psi * t24;
        double t651 = -86 * t24 * t36;
        double t652 = -46 * t153 * t24;
        double t653 = -410 * t24 * t3;
        double t654 = -400 * psi * t24 * t3;
        double t655 = 230 * t24 * t3 * t36;
        double t656 = 100 * t153 * t24 * t3;
        double t657 = 498 * t161 * t24;
        double t658 = 34 * psi * t161 * t24;
        double t659 = -242 * t161 * t24 * t36;
        double t660 = 22 * t153 * t161 * t24;
        double t661 = -198 * t167 * t24;
        double t662 = 192 * psi * t167 * t24;
        double t663 = 114 * t167 * t24 * t36;
        double t664 = -108 * t153 * t167 * t24;
        double t665 = -78 * t22 * t24;
        double t666 = -392 * psi * t22 * t24;
        double t667 = 200 * t22 * t24 * t36;
        double t668 = 130 * t153 * t22 * t24;
        double t669 = 298 * t22 * t24 * t3;
        double t670 = 952 * psi * t22 * t24 * t3;
        double t671 = -470 * t22 * t24 * t3 * t36;
        double t672 = -380 * t153 * t22 * t24 * t3;
        double t673 = -370 * t161 * t22 * t24;
        double t674 = -392 * psi * t161 * t22 * t24;
        double t675 = 288 * t161 * t22 * t24 * t36;
        double t676 = 214 * t153 * t161 * t22 * t24;
        double t678 = -200 * psi * t167 * t22 * t24;
        double t679 = -50 * t167 * t22 * t24 * t36;
        double t680 = 100 * t153 * t167 * t22 * t24;
        double t681 = 204 * psi * t24 * t73;
        double t682 = -132 * t24 * t36 * t73;
        double t683 = -72 * t153 * t24 * t73;
        double t684 = -568 * psi * t24 * t3 * t73;
        double t685 = 328 * t24 * t3 * t36 * t73;
        double t686 = 240 * t153 * t24 * t3 * t73;
        double t687 = 380 * psi * t161 * t24 * t73;
        double t688 = -180 * t161 * t24 * t36 * t73;
        double t689 = -200 * t153 * t161 * t24 * t73;
        double t690 = -1020 * t24 * t5;
        double t691 = -918 * psi * t24 * t5;
        double t692 = 980 * t24 * t36 * t5;
        double t693 = 78 * t153 * t24 * t5;
        double t694 = 16 * t159 * t24 * t5;
        double t695 = 2504 * t24 * t3 * t5;
        double t696 = -168 * psi * t24 * t3 * t5;
        double t697 = -1236 * t24 * t3 * t36 * t5;
        double t698 = 112 * t153 * t24 * t3 * t5;
        double t699 = -60 * t159 * t24 * t3 * t5;
        double t700 = -1500 * t161 * t24 * t5;
        double t701 = 1894 * psi * t161 * t24 * t5;
        double t702 = 40 * t161 * t24 * t36 * t5;
        double t703 = -470 * t153 * t161 * t24 * t5;
        double t704 = 36 * t159 * t161 * t24 * t5;
        double t705 = 716 * t22 * t24 * t5;
        double t706 = 2464 * psi * t22 * t24 * t5;
        double t707 = -2192 * t22 * t24 * t36 * t5;
        double t708 = -264 * t153 * t22 * t24 * t5;
        double t709 = -4 * t159 * t22 * t24 * t5;
        double t710 = -1800 * t22 * t24 * t3 * t5;
        double t711 = -2032 * psi * t22 * t24 * t3 * t5;
        double t712 = 2596 * t22 * t24 * t3 * t36 * t5;
        double t713 = 264 * t153 * t22 * t24 * t3 * t5;
        double t714 = 12 * t159 * t22 * t24 * t3 * t5;
        double t715 = 1100 * t161 * t22 * t24 * t5;
        double t716 = -1840 * psi * t161 * t22 * t24 * t5;
        double t717 = 380 * t161 * t22 * t24 * t36 * t5;
        double t719 = -1472 * psi * t24 * t5 * t73;
        double t720 = 1376 * t24 * t36 * t5 * t73;
        double t721 = 96 * t153 * t24 * t5 * t73;
        double t722 = 2080 * psi * t24 * t3 * t5 * t73;
        double t723 = -160 * t153 * t24 * t3 * t5 * t73;
        double t724 = 3048 * t24 * t266;
        double t725 = -528 * psi * t24 * t266;
        double t726 = -1800 * t24 * t266 * t36;
        double t727 = 384 * t153 * t24 * t266;
        double t728 = -48 * t159 * t24 * t266;
        double t729 = -3656 * t24 * t266 * t3;
        double t730 = 5488 * psi * t24 * t266 * t3;
        double t731 = -1328 * t24 * t266 * t3 * t36;
        double t732 = -576 * t153 * t24 * t266 * t3;
        double t733 = 72 * t159 * t24 * t266 * t3;
        double t734 = -2120 * t22 * t24 * t266;
        double t735 = -2624 * psi * t22 * t24 * t266;
        double t736 = 4472 * t22 * t24 * t266 * t36;
        double t737 = -616 * t153 * t22 * t24 * t266;
        double t738 = 8 * t159 * t22 * t24 * t266;
        double t739 = 2600 * t22 * t24 * t266 * t3;
        double t740 = -4960 * psi * t22 * t24 * t266 * t3;
        double t741 = 2120 * t22 * t24 * t266 * t3 * t36;
        double t742 = 240 * t153 * t22 * t24 * t266 * t3;
        double t743 = 2640 * psi * t24 * t266 * t73;
        double t744 = -3120 * t24 * t266 * t36 * t73;
        double t745 = 480 * t153 * t24 * t266 * t73;
        double t746 = -2896 * t24 * t298;
        double t747 = 4936 * psi * t24 * t298;
        double t748 = -1936 * t24 * t298 * t36;
        double t749 = -136 * t153 * t24 * t298;
        double t750 = 32 * t159 * t24 * t298;
        double t751 = 2000 * t22 * t24 * t298;
        double t752 = -4160 * psi * t22 * t24 * t298;
        double t753 = 2320 * t22 * t24 * t298 * t36;
        double t754 = -160 * t153 * t22 * t24 * t298;
        double t755 = -126 * t100;
        double t756 = -548 * psi * t100;
        double t757 = 210 * t100 * t36;
        double t758 = 296 * t100 * t153;
        double t759 = 498 * t100 * t3;
        double t760 = 1416 * psi * t100 * t3;
        double t761 = -590 * t100 * t3 * t36;
        double t762 = -844 * t100 * t153 * t3;
        double t763 = -642 * t100 * t161;
        double t764 = -708 * psi * t100 * t161;
        double t765 = 630 * t100 * t161 * t36;
        double t766 = 408 * t100 * t153 * t161;
        double t767 = 270 * t100 * t167;
        double t768 = -240 * psi * t100 * t167;
        double t769 = -330 * t100 * t167 * t36;
        double t770 = 300 * t100 * t153 * t167;
        double t771 = 564 * psi * t100 * t22;
        double t772 = -204 * t100 * t22 * t36;
        double t773 = -360 * t100 * t153 * t22;
        double t774 = -1624 * psi * t100 * t22 * t3;
        double t775 = 424 * t100 * t22 * t3 * t36;
        double t776 = 1200 * t100 * t153 * t22 * t3;
        double t777 = 1140 * psi * t100 * t161 * t22;
        double t778 = -140 * t100 * t161 * t22 * t36;
        double t779 = -1000 * t100 * t153 * t161 * t22;
        double t780 = 1140 * t100 * t5;
        double t781 = 3416 * psi * t100 * t5;
        double t782 = -2596 * t100 * t36 * t5;
        double t783 = -1088 * t100 * t153 * t5;
        double t784 = -8 * t100 * t159 * t5;
        double t785 = -2952 * t100 * t3 * t5;
        double t786 = -3216 * psi * t100 * t3 * t5;
        double t787 = 4048 * t100 * t3 * t36 * t5;
        double t788 = 944 * t100 * t153 * t3 * t5;
        double t789 = 24 * t100 * t159 * t3 * t5;
        double t790 = 1860 * t100 * t161 * t5;
        double t791 = -2280 * psi * t100 * t161 * t5;
        double t792 = -1020 * t100 * t161 * t36 * t5;
        double t793 = 1440 * t100 * t153 * t161 * t5;
        double t794 = -4016 * psi * t100 * t22 * t5;
        double t795 = 2624 * t100 * t22 * t36 * t5;
        double t796 = 1392 * t100 * t153 * t22 * t5;
        double t797 = 5840 * psi * t100 * t22 * t3 * t5;
        double t798 = -3520 * t100 * t22 * t3 * t36 * t5;
        double t799 = -2320 * t100 * t153 * t22 * t3 * t5;
        double t800 = -3336 * t100 * t266;
        double t801 = -3728 * psi * t100 * t266;
        double t802 = 6056 * t100 * t266 * t36;
        double t803 = -64 * t100 * t153 * t266;
        double t804 = 16 * t100 * t159 * t266;
        double t805 = 4200 * t100 * t266 * t3;
        double t806 = -6240 * psi * t100 * t266 * t3;
        double t807 = -120 * t100 * t266 * t3 * t36;
        double t808 = 7120 * psi * t100 * t22 * t266;
        double t809 = -6480 * t100 * t22 * t266 * t36;
        double t810 = -640 * t100 * t153 * t22 * t266;
        double t811 = 3120 * t100 * t298;
        double t812 = -5280 * psi * t100 * t298;
        double t813 = 1200 * t100 * t298 * t36;
        double t814 = 960 * t100 * t153 * t298;
        double t815 = 504 * psi * t491;
        double t816 = -72 * t36 * t491;
        double t817 = -1488 * psi * t3 * t491;
        double t818 = 48 * t3 * t36 * t491;
        double t819 = 1440 * t153 * t3 * t491;
        double t820 = 1080 * psi * t161 * t491;
        double t821 = 120 * t161 * t36 * t491;
        double t822 = -1200 * t153 * t161 * t491;
        double t823 = -3552 * psi * t491 * t5;
        double t824 = 1536 * t36 * t491 * t5;
        double t825 = 2016 * t153 * t491 * t5;
        double t826 = 5280 * psi * t3 * t491 * t5;
        double t827 = -1920 * t3 * t36 * t491 * t5;
        double t828 = -3360 * t153 * t3 * t491 * t5;
        double t829 = 6240 * psi * t266 * t491;
        double t830 = -1920 * t153 * t266 * t491;
        double t831 =
            -24 + t222 + t237 + t292 + t394 + t483 + t494 + t508 + t512 + t513 + t514 +
            t515 + t516 + t517 + t518 + t519 + t520 + t521 + t522 + t523 + t524 + t525 +
            t526 + t527 + t528 + t529 + t530 + t531 + t532 + t533 + t534 + t535 + t536 +
            t537 + t538 + t539 + t540 + t541 + t542 + t543 + t544 + t545 + t546 + t547 +
            t548 + t549 + t550 + t551 + t552 + t553 + t554 + t555 + t556 + t557 + t558 +
            t559 + t560 + t561 + t562 + t563 + t564 + t565 + t566 + t567 + t568 + t569 +
            t570 + t571 + t572 + t573 + t574 + t575 + t576 + t577 + t578 + t579 + t580 +
            t581 + t582 + t583 + t584 + t585 + t586 + t587 + t588 + t589 + t590 + t591 +
            t592 + t593 + t594 + t595 + t596 + t597 + t598 + t599 + t600 + t601 + t602 +
            t603 + t604 + t605 + t606 + t607 + t608 + t609 + t610 + t611 + t612 + t613 +
            t614 + t615 + t616 + t617 + t618 + t619 + t620 + t621 + t622 + t623 + t624 +
            t625 + t626 + t627 + t628 + t629 + t630 + t631 + t632 + t633 + t634 + t635 +
            t636 + t637 + t638 + t639 + t640 + t641 + t642 + t643 + t644 + t645 + t646 +
            t647 + t648 + t649 + t650 + t651 + t652 + t653 + t654 + t655 + t656 + t657 +
            t658 + t659 + t660 + t661 + t662 + t663 + t664 + t665 + t666 + t667 + t668 +
            t669 + t670 + t671 + t672 + t673 + t674 + t675 + t676 + t677 + t678 + t679 +
            t680 + t681 + t682 + t683 + t684 + t685 + t686 + t687 + t688 + t689 + t690 +
            t691 + t692 + t693 + t694 + t695 + t696 + t697 + t698 + t699 + t700 + t701 +
            t702 + t703 + t704 + t705 + t706 + t707 + t708 + t709 + t710 + t711 + t712 +
            t713 + t714 + t715 + t716 + t717 + t718 + t719 + t720 + t721 + t722 + t723 +
            t724 + t725 + t726 + t727 + t728 + t729 + t730 + t731 + t732 + t733 + t734 +
            t735 + t736 + t737 + t738 + t739 + t740 + t741 + t742 + t743 + t744 + t745 +
            t746 + t747 + t748 + t749 + t750 + t751 + t752 + t753 + t754 + t755 + t756 +
            t757 + t758 + t759 + t760 + t761 + t762 + t763 + t764 + t765 + t766 + t767 +
            t768 + t769 + t770 + t771 + t772 + t773 + t774 + t775 + t776 + t777 + t778 +
            t779 + t780 + t781 + t782 + t783 + t784 + t785 + t786 + t787 + t788 + t789 +
            t790 + t791 + t792 + t793 + t794 + t795 + t796 + t797 + t798 + t799 + t800 +
            t801 + t802 + t803 + t804 + t805 + t806 + t807 + t808 + t809 + t810 + t811 +
            t812 + t813 + t814 + t815 + t816 + t817 + t818 + t819 + t820 + t821 + t822 +
            t823 + t824 + t825 + t826 + t827 + t828 + t829 + t830;
        double t833 = -6 * psi;
        double t834 = 8 * t36;
        double t835 = 44 * t3;
        double t836 = 4 * psi * t3;
        double t837 = -37 * t3 * t36;
        double t838 = -9 * t153 * t3;
        double t839 = -72 * t161;
        double t840 = 46 * psi * t161;
        double t841 = 12 * t161 * t36;
        double t842 = 6 * t153 * t161;
        double t843 = -60 * psi * t167;
        double t844 = 33 * t167 * t36;
        double t845 = -3 * t153 * t167;
        double t846 = 37 * psi * t22;
        double t847 = -19 * t22 * t36;
        double t848 = -78 * t22 * t3;
        double t849 = -119 * psi * t22 * t3;
        double t850 = 140 * t22 * t3 * t36;
        double t851 = 55 * t153 * t22 * t3;
        double t852 = 130 * t161 * t22;
        double t853 = 35 * psi * t161 * t22;
        double t854 = -153 * t161 * t22 * t36;
        double t855 = -4 * t153 * t161 * t22;
        double t856 = -66 * t167 * t22;
        double t857 = 95 * psi * t167 * t22;
        double t858 = -16 * t167 * t22 * t36;
        double t859 = -19 * t153 * t167 * t22;
        double t860 = -40 * psi * t73;
        double t861 = -6 * t36 * t73;
        double t862 = 34 * t3 * t73;
        double t863 = 160 * psi * t3 * t73;
        double t864 = -78 * t3 * t36 * t73;
        double t865 = -58 * t161 * t73;
        double t866 = -128 * psi * t161 * t73;
        double t867 = 142 * t161 * t36 * t73;
        double t868 = 30 * t167 * t73;
        double t869 = -10 * t167 * t36 * t73;
        double t870 = 12 * psi * t208;
        double t871 = 12 * t208 * t36;
        double t872 = -56 * psi * t208 * t3;
        double t873 = -8 * t208 * t3 * t36;
        double t874 = 60 * psi * t161 * t208;
        double t875 = -20 * t161 * t208 * t36;
        double t876 = 60 * t5;
        double t877 = 26 * psi * t5;
        double t878 = -50 * t36 * t5;
        double t879 = -36 * t153 * t5;
        double t880 = -224 * t3 * t5;
        double t881 = 112 * psi * t3 * t5;
        double t882 = 114 * t3 * t36 * t5;
        double t883 = -10 * t153 * t3 * t5;
        double t884 = 8 * t159 * t3 * t5;
        double t885 = 180 * t161 * t5;
        double t886 = -290 * psi * t161 * t5;
        double t887 = 96 * t161 * t36 * t5;
        double t888 = 30 * t153 * t161 * t5;
        double t889 = -16 * t159 * t161 * t5;
        double t890 = -104 * t22 * t5;
        double t891 = -230 * psi * t22 * t5;
        double t892 = 126 * t22 * t36 * t5;
        double t893 = 208 * t153 * t22 * t5;
        double t894 = 392 * t22 * t3 * t5;
        double t895 = 268 * psi * t22 * t3 * t5;
        double t896 = -606 * t22 * t3 * t36 * t5;
        double t897 = -46 * t153 * t22 * t3 * t5;
        double t898 = -8 * t159 * t22 * t3 * t5;
        double t899 = -320 * t161 * t22 * t5;
        double t900 = 370 * psi * t161 * t22 * t5;
        double t901 = 144 * t161 * t22 * t36 * t5;
        double t902 = -210 * t153 * t161 * t22 * t5;
        double t903 = 16 * t159 * t161 * t22 * t5;
        double t904 = 44 * t5 * t73;
        double t905 = 240 * psi * t5 * t73;
        double t906 = 84 * t36 * t5 * t73;
        double t907 = -368 * t153 * t5 * t73;
        double t908 = -168 * t3 * t5 * t73;
        double t909 = -472 * psi * t3 * t5 * t73;
        double t910 = 304 * t3 * t36 * t5 * t73;
        double t911 = 336 * t153 * t3 * t5 * t73;
        double t912 = 140 * t161 * t5 * t73;
        double t913 = -120 * psi * t161 * t5 * t73;
        double t914 = -180 * t161 * t36 * t5 * t73;
        double t915 = 160 * t153 * t161 * t5 * t73;
        double t916 = -64 * psi * t208 * t5;
        double t917 = -112 * t208 * t36 * t5;
        double t918 = 176 * t153 * t208 * t5;
        double t919 = 160 * psi * t208 * t3 * t5;
        double t920 = 80 * t208 * t3 * t36 * t5;
        double t921 = -240 * t153 * t208 * t3 * t5;
        double t922 = -144 * t266;
        double t923 = -8 * psi * t266;
        double t924 = 144 * t266 * t36;
        double t925 = 4 * t153 * t266;
        double t926 = 4 * t159 * t266;
        double t927 = 272 * t266 * t3;
        double t928 = -360 * psi * t266 * t3;
        double t929 = -56 * t266 * t3 * t36;
        double t930 = 164 * t153 * t266 * t3;
        double t931 = -20 * t159 * t266 * t3;
        double t932 = 248 * t22 * t266;
        double t933 = 412 * psi * t22 * t266;
        double t934 = -476 * t22 * t266 * t36;
        double t935 = -180 * t153 * t22 * t266;
        double t936 = -4 * t159 * t22 * t266;
        double t937 = -472 * t22 * t266 * t3;
        double t938 = 260 * psi * t22 * t266 * t3;
        double t939 = 820 * t22 * t266 * t3 * t36;
        double t940 = -628 * t153 * t22 * t266 * t3;
        double t941 = 20 * t159 * t22 * t266 * t3;
        double t942 = -104 * t266 * t73;
        double t943 = -400 * psi * t266 * t73;
        double t944 = -56 * t266 * t36 * t73;
        double t945 = 560 * t153 * t266 * t73;
        double t946 = 200 * t266 * t3 * t73;
        double t947 = -600 * t266 * t3 * t36 * t73;
        double t948 = 400 * t153 * t266 * t3 * t73;
        double t949 = 80 * psi * t208 * t266;
        double t950 = 240 * t208 * t266 * t36;
        double t951 = -320 * t153 * t208 * t266;
        double t952 = 112 * t298;
        double t953 = -40 * psi * t298;
        double t954 = -248 * t298 * t36;
        double t955 = 184 * t153 * t298;
        double t956 = -8 * t159 * t298;
        double t957 = -192 * t22 * t298;
        double t958 = -200 * psi * t22 * t298;
        double t959 = 952 * t22 * t298 * t36;
        double t960 = -568 * t153 * t22 * t298;
        double t961 = 8 * t159 * t22 * t298;
        double t962 = 80 * t298 * t73;
        double t963 = 160 * psi * t298 * t73;
        double t964 = -560 * t298 * t36 * t73;
        double t965 = 320 * t153 * t298 * t73;
        double t966 = 45 * psi * t24;
        double t967 = -46 * t24 * t36;
        double t968 = -35 * t153 * t24;
        double t969 = -190 * t24 * t3;
        double t970 = -87 * psi * t24 * t3;
        double t971 = 255 * t24 * t3 * t36;
        double t972 = 26 * t153 * t24 * t3;
        double t973 = 318 * t161 * t24;
        double t974 = -197 * psi * t161 * t24;
        double t975 = -108 * t161 * t24 * t36;
        double t976 = -3 * t153 * t161 * t24;
        double t977 = -162 * t167 * t24;
        double t978 = 351 * psi * t167 * t24;
        double t979 = -249 * t167 * t24 * t36;
        double t980 = 48 * t153 * t167 * t24;
        double t981 = -160 * psi * t22 * t24;
        double t982 = 51 * t22 * t24 * t36;
        double t983 = 141 * t153 * t22 * t24;
        double t984 = 170 * t22 * t24 * t3;
        double t985 = 616 * psi * t22 * t24 * t3;
        double t986 = -580 * t22 * t24 * t3 * t36;
        double t987 = -214 * t153 * t22 * t24 * t3;
        double t988 = -290 * t161 * t22 * t24;
        double t989 = -400 * psi * t161 * t22 * t24;
        double t990 = 711 * t161 * t22 * t24 * t36;
        double t991 = -15 * t153 * t161 * t22 * t24;
        double t992 = -280 * psi * t167 * t22 * t24;
        double t993 = 110 * t167 * t22 * t24 * t36;
        double t994 = 20 * t153 * t167 * t22 * t24;
        double t995 = 84 * psi * t24 * t73;
        double t996 = 36 * t24 * t36 * t73;
        double t997 = -120 * t153 * t24 * t73;
        double t998 = -392 * psi * t24 * t3 * t73;
        double t999 = 120 * t24 * t3 * t36 * t73;
        double t1000 = 272 * t153 * t24 * t3 * t73;
        double t1001 = 420 * psi * t161 * t24 * t73;
        double t1002 = -300 * t161 * t24 * t36 * t73;
        double t1003 = -120 * t153 * t161 * t24 * t73;
        double t1004 = -252 * t24 * t5;
        double t1005 = -198 * psi * t24 * t5;
        double t1006 = 174 * t24 * t36 * t5;
        double t1007 = 288 * t153 * t24 * t5;
        double t1008 = -12 * t159 * t24 * t5;
        double t1009 = 952 * t24 * t3 * t5;
        double t1010 = -388 * psi * t24 * t3 * t5;
        double t1011 = -476 * t24 * t3 * t36 * t5;
        double t1012 = -100 * t153 * t24 * t3 * t5;
        double t1013 = 12 * t159 * t24 * t3 * t5;
        double t1014 = -780 * t161 * t24 * t5;
        double t1015 = 1570 * psi * t161 * t24 * t5;
        double t1016 = -882 * t161 * t24 * t36 * t5;
        double t1017 = 60 * t153 * t161 * t24 * t5;
        double t1018 = 32 * t159 * t161 * t24 * t5;
        double t1019 = 220 * t22 * t24 * t5;
        double t1020 = 912 * psi * t22 * t24 * t5;
        double t1021 = -56 * t22 * t24 * t36 * t5;
        double t1022 = -1088 * t153 * t22 * t24 * t5;
        double t1023 = 12 * t159 * t22 * t24 * t5;
        double t1024 = -840 * t22 * t24 * t3 * t5;
        double t1025 = -1536 * psi * t22 * t24 * t3 * t5;
        double t1026 = 1700 * t22 * t24 * t3 * t36 * t5;
        double t1027 = 704 * t153 * t22 * t24 * t3 * t5;
        double t1028 = -28 * t159 * t22 * t24 * t3 * t5;
        double t1029 = 700 * t161 * t22 * t24 * t5;
        double t1030 = -1040 * psi * t161 * t22 * t24 * t5;
        double t1031 = -20 * t161 * t22 * t24 * t36 * t5;
        double t1032 = -448 * psi * t24 * t5 * t73;
        double t1033 = -480 * t24 * t36 * t5 * t73;
        double t1034 = 928 * t153 * t24 * t5 * t73;
        double t1035 = 1120 * psi * t24 * t3 * t5 * t73;
        double t1036 = -1120 * t153 * t24 * t3 * t5 * t73;
        double t1037 = 600 * t24 * t266;
        double t1038 = 60 * psi * t24 * t266;
        double t1039 = -256 * t24 * t266 * t36;
        double t1040 = -412 * t153 * t24 * t266;
        double t1041 = 8 * t159 * t24 * t266;
        double t1042 = -1144 * t24 * t266 * t3;
        double t1043 = 1892 * psi * t24 * t266 * t3;
        double t1044 = -404 * t24 * t266 * t3 * t36;
        double t1045 = -384 * t153 * t24 * t266 * t3;
        double t1046 = 40 * t159 * t24 * t266 * t3;
        double t1047 = -520 * t22 * t24 * t266;
        double t1048 = -1344 * psi * t22 * t24 * t266;
        double t1049 = 124 * t22 * t24 * t266 * t36;
        double t1050 = 1756 * t153 * t22 * t24 * t266;
        double t1051 = -16 * t159 * t22 * t24 * t266;
        double t1052 = 1000 * t22 * t24 * t266 * t3;
        double t1053 = -800 * psi * t22 * t24 * t266 * t3;
        double t1054 = -1400 * t22 * t24 * t266 * t3 * t36;
        double t1055 = 1200 * t153 * t22 * t24 * t266 * t3;
        double t1056 = 560 * psi * t24 * t266 * t73;
        double t1057 = 1200 * t24 * t266 * t36 * t73;
        double t1058 = -1760 * t153 * t24 * t266 * t73;
        double t1059 = -464 * t24 * t298;
        double t1060 = 312 * psi * t24 * t298;
        double t1061 = 760 * t24 * t298 * t36;
        double t1062 = -624 * t153 * t24 * t298;
        double t1063 = 16 * t159 * t24 * t298;
        double t1064 = 400 * t22 * t24 * t298;
        double t1065 = 320 * psi * t22 * t24 * t298;
        double t1066 = -1840 * t22 * t24 * t298 * t36;
        double t1067 = 1120 * t153 * t22 * t24 * t298;
        double t1068 = -136 * psi * t100;
        double t1069 = 126 * t100 * t36;
        double t1070 = 50 * t100 * t153;
        double t1071 = 204 * t100 * t3;
        double t1072 = 480 * psi * t100 * t3;
        double t1073 = -800 * t100 * t3 * t36;
        double t1074 = 100 * t100 * t153 * t3;
        double t1075 = -348 * t100 * t161;
        double t1076 = -120 * psi * t100 * t161;
        double t1077 = 678 * t100 * t161 * t36;
        double t1078 = -198 * t100 * t153 * t161;
        double t1079 = 180 * t100 * t167;
        double t1080 = -480 * psi * t100 * t167;
        double t1081 = 420 * t100 * t167 * t36;
        double t1082 = -120 * t100 * t153 * t167;
        double t1083 = 192 * psi * t100 * t22;
        double t1084 = -48 * t100 * t22 * t36;
        double t1085 = -144 * t100 * t153 * t22;
        double t1086 = -896 * psi * t100 * t22 * t3;
        double t1087 = 752 * t100 * t22 * t3 * t36;
        double t1088 = 144 * t100 * t153 * t22 * t3;
        double t1089 = 960 * psi * t100 * t161 * t22;
        double t1090 = -1120 * t100 * t161 * t22 * t36;
        double t1091 = 160 * t100 * t153 * t161 * t22;
        double t1092 = 264 * t100 * t5;
        double t1093 = 664 * psi * t100 * t5;
        double t1094 = -352 * t100 * t36 * t5;
        double t1095 = -600 * t100 * t153 * t5;
        double t1096 = 24 * t100 * t159 * t5;
        double t1097 = -1008 * t100 * t3 * t5;
        double t1098 = -576 * psi * t100 * t3 * t5;
        double t1099 = 1528 * t100 * t3 * t36 * t5;
        double t1100 = 112 * t100 * t153 * t3 * t5;
        double t1101 = -56 * t100 * t159 * t3 * t5;
        double t1102 = 840 * t100 * t161 * t5;
        double t1103 = -2040 * psi * t100 * t161 * t5;
        double t1104 = 1560 * t100 * t161 * t36 * t5;
        double t1105 = -360 * t100 * t153 * t161 * t5;
        double t1106 = -1024 * psi * t100 * t22 * t5;
        double t1107 = -272 * t100 * t22 * t36 * t5;
        double t1108 = 1296 * t100 * t153 * t22 * t5;
        double t1109 = 2560 * psi * t100 * t22 * t3 * t5;
        double t1110 = -1520 * t100 * t22 * t3 * t36 * t5;
        double t1111 = -1040 * t100 * t153 * t22 * t3 * t5;
        double t1112 = -624 * t100 * t266;
        double t1113 = -544 * psi * t100 * t266;
        double t1114 = -136 * t100 * t266 * t36;
        double t1115 = 1336 * t100 * t153 * t266;
        double t1116 = -32 * t100 * t159 * t266;
        double t1117 = 1200 * t100 * t266 * t3;
        double t1118 = -2400 * psi * t100 * t266 * t3;
        double t1119 = 1200 * t100 * t266 * t3 * t36;
        double t1120 = 1280 * psi * t100 * t22 * t266;
        double t1121 = 1440 * t100 * t22 * t266 * t36;
        double t1122 = -2720 * t100 * t153 * t22 * t266;
        double t1123 = 480 * t100 * t298;
        double t1124 = -480 * psi * t100 * t298;
        double t1125 = -480 * t100 * t298 * t36;
        double t1126 = 480 * t100 * t153 * t298;
        double t1127 = 144 * psi * t491;
        double t1128 = -144 * t36 * t491;
        double t1129 = -672 * psi * t3 * t491;
        double t1130 = 960 * t3 * t36 * t491;
        double t1131 = -288 * t153 * t3 * t491;
        double t1132 = 720 * psi * t161 * t491;
        double t1133 = -1200 * t161 * t36 * t491;
        double t1134 = 480 * t153 * t161 * t491;
        double t1135 = -768 * psi * t491 * t5;
        double t1136 = 480 * t36 * t491 * t5;
        double t1137 = 288 * t153 * t491 * t5;
        double t1138 = 1920 * psi * t3 * t491 * t5;
        double t1139 = -2400 * t3 * t36 * t491 * t5;
        double t1140 = 480 * t153 * t3 * t491 * t5;
        double t1141 = 960 * psi * t266 * t491;
        double t1142 = -960 * t153 * t266 * t491;
        double t1143 =
            -8 + t1000 + t1001 + t1002 + t1003 + t1004 + t1005 + t1006 + t1007 + t1008 +
            t1009 + t101 + t1010 + t1011 + t1012 + t1013 + t1014 + t1015 + t1016 + t1017 +
            t1018 + t1019 + t1020 + t1021 + t1022 + t1023 + t1024 + t1025 + t1026 +
            t1027 + t1028 + t1029 + t1030 + t1031 + t1032 + t1033 + t1034 + t1035 +
            t1036 + t1037 + t1038 + t1039 + t1040 + t1041 + t1042 + t1043 + t1044 +
            t1045 + t1046 + t1047 + t1048 + t1049 + t1050 + t1051 + t1052 + t1053 +
            t1054 + t1055 + t1056 + t1057 + t1058 + t1059 + t1060 + t1061 + t1062 +
            t1063 + t1064 + t1065 + t1066 + t1067 + t1068 + t1069 + t1070 + t1071 +
            t1072 + t1073 + t1074 + t1075 + t1076 + t1077 + t1078 + t1079 + t1080 +
            t1081 + t1082 + t1083 + t1084 + t1085 + t1086 + t1087 + t1088 + t1089 +
            t1090 + t1091 + t1092 + t1093 + t1094 + t1095 + t1096 + t1097 + t1098 +
            t1099 + t1100 + t1101 + t1102 + t1103 + t1104 + t1105 + t1106 + t1107 +
            t1108 + t1109 + t1110 + t1111 + t1112 + t1113 + t1114 + t1115 + t1116 +
            t1117 + t1118 + t1119 + t1120 + t1121 + t1122 + t1123 + t1124 + t1125 +
            t1126 + t1127 + t1128 + t1129 + t1130 + t1131 + t1132 + t1133 + t1134 +
            t1135 + t1136 + t1137 + t1138 + t1139 + t1140 + t1141 + t1142 + t154 + t160 +
            t166 + t172 + t176 + t181 + t186 + t191 + t195 + t199 + t203 + t207 + t211 +
            t214 + t217 + t317 + t322 + t327 + t332 + t337 + t342 + t347 + t431 + t436 +
            t441 + t522 + t551 + t67 + t677 + t718 + t74 + t833 + t834 + t835 + t836 +
            t837 + t838 + t839 + t840 + t841 + t842 + t843 + t844 + t845 + t846 + t847 +
            t848 + t849 + t850 + t851 + t852 + t853 + t854 + t855 + t856 + t857 + t858 +
            t859 + t86 + t860 + t861 + t862 + t863 + t864 + t865 + t866 + t867 + t868 +
            t869 + t870 + t871 + t872 + t873 + t874 + t875 + t876 + t877 + t878 + t879 +
            t880 + t881 + t882 + t883 + t884 + t885 + t886 + t887 + t888 + t889 + t890 +
            t891 + t892 + t893 + t894 + t895 + t896 + t897 + t898 + t899 + t900 + t901 +
            t902 + t903 + t904 + t905 + t906 + t907 + t908 + t909 + t91 + t910 + t911 +
            t912 + t913 + t914 + t915 + t916 + t917 + t918 + t919 + t920 + t921 + t922 +
            t923 + t924 + t925 + t926 + t927 + t928 + t929 + t930 + t931 + t932 + t933 +
            t934 + t935 + t936 + t937 + t938 + t939 + t940 + t941 + t942 + t943 + t944 +
            t945 + t946 + t947 + t948 + t949 + t950 + t951 + t952 + t953 + t954 + t955 +
            t956 + t957 + t958 + t959 + t960 + t961 + t962 + t963 + t964 + t965 + t966 +
            t967 + t968 + t969 + t970 + t971 + t972 + t973 + t974 + t975 + t976 + t977 +
            t978 + t979 + t980 + t981 + t982 + t983 + t984 + t985 + t986 + t987 + t988 +
            t989 + t990 + t991 + t992 + t993 + t994 + t995 + t996 + t997 + t998 + t999;
        double t1144 = (3 * t1143 * t511) / 2.;
        c[0] = -1 + t3 + t9;
        c[1] = -(psi * (-1 + 2 * t3 + 3 * t5));
        c[2] = 1 + t11 + t12 + t13 + t14 + 3 * psi * t5;
        c[3] = t18 + psi * t5;
        c[4] = t21;
        c[5] = t29 * (-1 + t11 + 2 * psi * t22 + 4 * psi * t24 + t3 - psi * t3 -
                      2 * psi * t5 + t9);
        c[6] = -(psi * (-1 + 2 * t22 + 3 * t24));
        c[7] = t29 * (1 + t12 + t14 - 2 * psi * t24 + psi * t3 + 2 * t22 * t36 +
                      t3 * t36 + t37 + t44 + 2 * psi * t5 + t36 * t5);
        c[8] = t18 + psi * t24;
        c[9] = t21;
        c[10] = (psi * t50 * t52 * t62) / 2.;
        c[11] =
            (t62 *
             (-8 + 2 * psi + 36 * psi * t100 + t101 - 9 * psi * t22 - 21 * psi * t24 +
              30 * psi * t22 * t24 + 12 * t3 + 60 * t100 * t3 - 60 * psi * t100 * t3 -
              22 * t22 * t3 + 11 * psi * t22 * t3 - 54 * t24 * t3 + 27 * psi * t24 * t3 +
              50 * t22 * t24 * t3 - 50 * psi * t22 * t24 * t3 + t22 * t36 + 3 * t3 * t36 -
              3 * t22 * t3 * t36 - 6 * t24 * t3 * t36 + t37 + t44 + 28 * t5 -
              10 * psi * t5 + 120 * t100 * t5 - 120 * psi * t100 * t5 - 48 * t22 * t5 +
              34 * psi * t22 * t5 - 116 * t24 * t5 + 78 * psi * t24 * t5 +
              100 * t22 * t24 * t5 - 100 * psi * t22 * t24 * t5 + 2 * t36 * t5 -
              2 * t22 * t36 * t5 - 4 * t24 * t36 * t5 + t67 + 6 * psi * t73 +
              10 * t3 * t73 - 10 * psi * t3 * t73 + 20 * t5 * t73 - 20 * psi * t5 * t73 +
              t74 + t86 + t91)) /
            2.;
        c[12] =
            1 + 3 * t17 * t24 + (t36 * t50 * t52 * t62) / 2. +
            t62 * (-3 * psi + t13 - 8 * t22 + 10 * psi * t22 + 7 * psi * t24 +
                   18 * t22 * t24 - 18 * psi * t22 * t24 + 12 * t22 * t3 -
                   14 * psi * t22 * t3 - 9 * psi * t24 * t3 - 30 * t22 * t24 * t3 +
                   30 * psi * t22 * t24 * t3 + 12 * psi * t5 + 28 * t22 * t5 -
                   36 * psi * t22 * t5 - 26 * psi * t24 * t5 - 60 * t22 * t24 * t5 +
                   60 * psi * t22 * t24 * t5 + 6 * t73 - 6 * psi * t73 - 10 * t3 * t73 +
                   10 * psi * t3 * t73 - 20 * t5 * t73 + 20 * psi * t5 * t73);
        c[13] = -(t17 * t22) + (psi *
                                (3 - 2 * t22 - 7 * t24 - 3 * t3 + 2 * t22 * t3 +
                                 9 * t24 * t3 - 12 * t5 + 8 * t22 * t5 + 26 * t24 * t5) *
                                t62) /
                                   2.;
        c[14] = (1 + t11) * t24;
        c[15] = 0;
        c[16] = -1 + t1144 + t511 * t831;
        c[17] = 1 - 3 * t1143 * t511 - (3 * t511 * t831) / 2.;
        c[18] = t1144;
        c[19] = (t511 * t831) / 2.;

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
        double t3 = alpha[0];
        double t5 = alpha[2];
        double t17 = -1 + psi;
        double t11 = -psi;
        double t22 = alpha[1];
        double t9 = 2 * t5;
        double t24 = alpha[3];
        double t28 = 1 + psi;
        double t29 = std::pow(t28, -1);
        double t12 = -2 * t3;
        double t36 = std::pow(psi, 2);
        double t14 = -3 * t5;
        double t18 = -(t17 * t3);
        double t21 = -(t17 * t5);
        double t53 = -6 * t3;
        double t54 = -3 * t22;
        double t55 = 5 * t22 * t3;
        double t56 = -14 * t5;
        double t57 = 10 * t22 * t5;
        double t58 = -9 * t24;
        double t59 = 15 * t24 * t3;
        double t60 = 30 * t24 * t5;
        double t61 = 4 + t53 + t54 + t55 + t56 + t57 + t58 + t59 + t60;
        double t62 = std::pow(t61, -1);
        double t37 = -t36;
        double t73 = std::pow(t22, 2);
        double t44 = 2 * t24 * t36;
        double t100 = std::pow(t24, 2);
        double t49 = 3 * t3;
        double t50 = -1 + t49 + t9;
        double t51 = 2 * t24;
        double t52 = -1 + t22 + t51;
        double t13 = 3 * psi * t3;
        double t153 = std::pow(psi, 3);
        double t161 = std::pow(t3, 2);
        double t159 = std::pow(psi, 4);
        double t167 = std::pow(t3, 3);
        double t208 = std::pow(t22, 3);
        double t266 = std::pow(t5, 2);
        double t298 = std::pow(t5, 3);
        double t491 = std::pow(t24, 3);
        double t222 = -6 * t159 * t5;
        double t237 = 6 * t159 * t22 * t5;
        double t292 = -960 * psi * t266 * t3 * t73;
        double t394 = -1920 * t24 * t3 * t36 * t5 * t73;
        double t483 = 2160 * t100 * t153 * t266 * t3;
        double t494 = -432 * t153 * t491;
        double t508 = -4320 * t266 * t36 * t491;
        double t151 = -27 * psi;
        double t152 = 11 * t36;
        double t154 = 6 * t153;
        double t155 = 128 * t3;
        double t156 = 36 * psi * t3;
        double t157 = -31 * t3 * t36;
        double t158 = -11 * t153 * t3;
        double t160 = -2 * t159 * t3;
        double t162 = -168 * t161;
        double t163 = 71 * psi * t161;
        double t164 = -3 * t161 * t36;
        double t165 = 14 * t153 * t161;
        double t166 = 8 * t159 * t161;
        double t168 = 72 * t167;
        double t169 = -96 * psi * t167;
        double t170 = 39 * t167 * t36;
        double t171 = -9 * t153 * t167;
        double t172 = -6 * t159 * t167;
        double t173 = 48 * t22;
        double t174 = 102 * psi * t22;
        double t175 = -48 * t22 * t36;
        double t176 = -32 * t153 * t22;
        double t177 = -200 * t22 * t3;
        double t178 = -239 * psi * t22 * t3;
        double t179 = 180 * t22 * t3 * t36;
        double t180 = 57 * t153 * t22 * t3;
        double t181 = 2 * t159 * t22 * t3;
        double t182 = 272 * t161 * t22;
        double t183 = 14 * psi * t161 * t22;
        double t184 = -136 * t161 * t22 * t36;
        double t185 = -12 * t153 * t161 * t22;
        double t186 = -8 * t159 * t161 * t22;
        double t187 = -120 * t167 * t22;
        double t188 = 171 * psi * t167 * t22;
        double t189 = -44 * t167 * t22 * t36;
        double t190 = -13 * t153 * t167 * t22;
        double t191 = 6 * t159 * t167 * t22;
        double t192 = -18 * t73;
        double t193 = -108 * psi * t73;
        double t194 = 46 * t36 * t73;
        double t195 = 52 * t153 * t73;
        double t196 = 78 * t3 * t73;
        double t197 = 312 * psi * t3 * t73;
        double t198 = -194 * t3 * t36 * t73;
        double t199 = -116 * t153 * t3 * t73;
        double t200 = -110 * t161 * t73;
        double t201 = -172 * psi * t161 * t73;
        double t202 = 186 * t161 * t36 * t73;
        double t203 = 44 * t153 * t161 * t73;
        double t204 = 50 * t167 * t73;
        double t205 = -80 * psi * t167 * t73;
        double t206 = 10 * t167 * t36 * t73;
        double t207 = 20 * t153 * t167 * t73;
        double t209 = 36 * psi * t208;
        double t210 = -12 * t208 * t36;
        double t211 = -24 * t153 * t208;
        double t212 = -120 * psi * t208 * t3;
        double t213 = 56 * t208 * t3 * t36;
        double t214 = 64 * t153 * t208 * t3;
        double t215 = 100 * psi * t161 * t208;
        double t216 = -60 * t161 * t208 * t36;
        double t217 = -40 * t153 * t161 * t208;
        double t218 = 288 * t5;
        double t219 = 75 * psi * t5;
        double t220 = -104 * t36 * t5;
        double t221 = -37 * t153 * t5;
        double t223 = -752 * t3 * t5;
        double t224 = 340 * psi * t3 * t5;
        double t225 = 42 * t3 * t36 * t5;
        double t226 = 50 * t153 * t3 * t5;
        double t227 = 32 * t159 * t3 * t5;
        double t228 = 480 * t161 * t5;
        double t229 = -667 * psi * t161 * t5;
        double t230 = 258 * t161 * t36 * t5;
        double t231 = -37 * t153 * t161 * t5;
        double t232 = -34 * t159 * t161 * t5;
        double t233 = -424 * t22 * t5;
        double t234 = -539 * psi * t22 * t5;
        double t235 = 440 * t22 * t36 * t5;
        double t236 = 157 * t153 * t22 * t5;
        double t238 = 1152 * t22 * t3 * t5;
        double t239 = 104 * psi * t22 * t3 * t5;
        double t240 = -678 * t22 * t3 * t36 * t5;
        double t241 = -66 * t153 * t22 * t3 * t5;
        double t242 = -32 * t159 * t22 * t3 * t5;
        double t243 = -760 * t161 * t22 * t5;
        double t244 = 1103 * psi * t161 * t22 * t5;
        double t245 = -278 * t161 * t22 * t36 * t5;
        double t246 = -99 * t153 * t161 * t22 * t5;
        double t247 = 34 * t159 * t161 * t22 * t5;
        double t248 = 156 * t5 * t73;
        double t249 = 672 * psi * t5 * t73;
        double t250 = -420 * t36 * t5 * t73;
        double t251 = -264 * t153 * t5 * t73;
        double t252 = -440 * t3 * t5 * t73;
        double t253 = -768 * psi * t3 * t5 * t73;
        double t254 = 808 * t3 * t36 * t5 * t73;
        double t255 = 208 * t153 * t3 * t5 * t73;
        double t256 = 300 * t161 * t5 * t73;
        double t257 = -480 * psi * t161 * t5 * t73;
        double t258 = 60 * t161 * t36 * t5 * t73;
        double t259 = 120 * t153 * t161 * t5 * t73;
        double t260 = -240 * psi * t208 * t5;
        double t261 = 112 * t208 * t36 * t5;
        double t262 = 128 * t153 * t208 * t5;
        double t263 = 400 * psi * t208 * t3 * t5;
        double t264 = -240 * t208 * t3 * t36 * t5;
        double t265 = -160 * t153 * t208 * t3 * t5;
        double t267 = -840 * t266;
        double t268 = 408 * psi * t266;
        double t269 = 112 * t266 * t36;
        double t270 = 32 * t153 * t266;
        double t271 = 24 * t159 * t266;
        double t272 = 1064 * t266 * t3;
        double t273 = -1544 * psi * t266 * t3;
        double t274 = 556 * t266 * t3 * t36;
        double t275 = -20 * t153 * t266 * t3;
        double t276 = -56 * t159 * t266 * t3;
        double t277 = 1216 * t22 * t266;
        double t278 = 148 * psi * t22 * t266;
        double t279 = -828 * t22 * t266 * t36;
        double t280 = -72 * t153 * t22 * t266;
        double t281 = -24 * t159 * t22 * t266;
        double t282 = -1600 * t22 * t266 * t3;
        double t283 = 2380 * psi * t22 * t266 * t3;
        double t284 = -576 * t22 * t266 * t3 * t36;
        double t285 = -260 * t153 * t22 * t266 * t3;
        double t286 = 56 * t159 * t22 * t266 * t3;
        double t287 = -440 * t266 * t73;
        double t288 = -848 * psi * t266 * t73;
        double t289 = 872 * t266 * t36 * t73;
        double t290 = 240 * t153 * t266 * t73;
        double t291 = 600 * t266 * t3 * t73;
        double t293 = 120 * t266 * t3 * t36 * t73;
        double t294 = 240 * t153 * t266 * t3 * t73;
        double t295 = 400 * psi * t208 * t266;
        double t296 = -240 * t208 * t266 * t36;
        double t297 = -160 * t153 * t208 * t266;
        double t299 = 784 * t298;
        double t300 = -1188 * psi * t298;
        double t301 = 392 * t298 * t36;
        double t302 = 36 * t153 * t298;
        double t303 = -24 * t159 * t298;
        double t304 = -1120 * t22 * t298;
        double t305 = 1716 * psi * t22 * t298;
        double t306 = -392 * t22 * t298 * t36;
        double t307 = -228 * t153 * t22 * t298;
        double t308 = 24 * t159 * t22 * t298;
        double t309 = 400 * t298 * t73;
        double t310 = -640 * psi * t298 * t73;
        double t311 = 80 * t298 * t36 * t73;
        double t312 = 160 * t153 * t298 * t73;
        double t313 = 144 * t24;
        double t314 = 235 * psi * t24;
        double t315 = -132 * t24 * t36;
        double t316 = -81 * t153 * t24;
        double t317 = 2 * t159 * t24;
        double t318 = -600 * t24 * t3;
        double t319 = -487 * psi * t24 * t3;
        double t320 = 485 * t24 * t3 * t36;
        double t321 = 126 * t153 * t24 * t3;
        double t322 = -4 * t159 * t24 * t3;
        double t323 = 816 * t161 * t24;
        double t324 = -163 * psi * t161 * t24;
        double t325 = -350 * t161 * t24 * t36;
        double t326 = 19 * t153 * t161 * t24;
        double t327 = -10 * t159 * t161 * t24;
        double t328 = -360 * t167 * t24;
        double t329 = 543 * psi * t167 * t24;
        double t330 = -135 * t167 * t24 * t36;
        double t331 = -60 * t153 * t167 * t24;
        double t332 = 12 * t159 * t167 * t24;
        double t333 = -108 * t22 * t24;
        double t334 = -552 * psi * t22 * t24;
        double t335 = 251 * t22 * t24 * t36;
        double t336 = 271 * t153 * t22 * t24;
        double t337 = -2 * t159 * t22 * t24;
        double t338 = 468 * t22 * t24 * t3;
        double t339 = 1568 * psi * t22 * t24 * t3;
        double t340 = -1050 * t22 * t24 * t3 * t36;
        double t341 = -594 * t153 * t22 * t24 * t3;
        double t342 = 8 * t159 * t22 * t24 * t3;
        double t343 = -660 * t161 * t22 * t24;
        double t344 = -792 * psi * t161 * t22 * t24;
        double t345 = 999 * t161 * t22 * t24 * t36;
        double t346 = 199 * t153 * t161 * t22 * t24;
        double t347 = -6 * t159 * t161 * t22 * t24;
        double t348 = 300 * t167 * t22 * t24;
        double t349 = -480 * psi * t167 * t22 * t24;
        double t350 = 60 * t167 * t22 * t24 * t36;
        double t351 = 120 * t153 * t167 * t22 * t24;
        double t352 = 288 * psi * t24 * t73;
        double t353 = -96 * t24 * t36 * t73;
        double t354 = -192 * t153 * t24 * t73;
        double t355 = -960 * psi * t24 * t3 * t73;
        double t356 = 448 * t24 * t3 * t36 * t73;
        double t357 = 512 * t153 * t24 * t3 * t73;
        double t358 = 800 * psi * t161 * t24 * t73;
        double t359 = -480 * t161 * t24 * t36 * t73;
        double t360 = -320 * t153 * t161 * t24 * t73;
        double t361 = -1272 * t24 * t5;
        double t362 = -1116 * psi * t24 * t5;
        double t363 = 1154 * t24 * t36 * t5;
        double t364 = 366 * t153 * t24 * t5;
        double t365 = 4 * t159 * t24 * t5;
        double t366 = 3456 * t24 * t3 * t5;
        double t367 = -556 * psi * t24 * t3 * t5;
        double t368 = -1712 * t24 * t3 * t36 * t5;
        double t369 = 12 * t153 * t24 * t3 * t5;
        double t370 = -48 * t159 * t24 * t3 * t5;
        double t371 = -2280 * t161 * t24 * t5;
        double t372 = 3464 * psi * t161 * t24 * t5;
        double t373 = -842 * t161 * t24 * t36 * t5;
        double t374 = -410 * t153 * t161 * t24 * t5;
        double t375 = 68 * t159 * t161 * t24 * t5;
        double t376 = 936 * t22 * t24 * t5;
        double t377 = 3376 * psi * t22 * t24 * t5;
        double t378 = -2248 * t22 * t24 * t36 * t5;
        double t379 = -1352 * t153 * t22 * t24 * t5;
        double t380 = 8 * t159 * t22 * t24 * t5;
        double t381 = -2640 * t22 * t24 * t3 * t5;
        double t382 = -3568 * psi * t22 * t24 * t3 * t5;
        double t383 = 4296 * t22 * t24 * t3 * t36 * t5;
        double t384 = 968 * t153 * t22 * t24 * t3 * t5;
        double t385 = -16 * t159 * t22 * t24 * t3 * t5;
        double t386 = 1800 * t161 * t22 * t24 * t5;
        double t387 = -2880 * psi * t161 * t22 * t24 * t5;
        double t388 = 360 * t161 * t22 * t24 * t36 * t5;
        double t389 = 720 * t153 * t161 * t22 * t24 * t5;
        double t390 = -1920 * psi * t24 * t5 * t73;
        double t391 = 896 * t24 * t36 * t5 * t73;
        double t392 = 1024 * t153 * t24 * t5 * t73;
        double t393 = 3200 * psi * t24 * t3 * t5 * t73;
        double t395 = -1280 * t153 * t24 * t3 * t5 * t73;
        double t396 = 3648 * t24 * t266;
        double t397 = -468 * psi * t24 * t266;
        double t398 = -2056 * t24 * t266 * t36;
        double t399 = -28 * t153 * t24 * t266;
        double t400 = -40 * t159 * t24 * t266;
        double t401 = -4800 * t24 * t266 * t3;
        double t402 = 7380 * psi * t24 * t266 * t3;
        double t403 = -1732 * t24 * t266 * t3 * t36;
        double t404 = -960 * t153 * t24 * t266 * t3;
        double t405 = 112 * t159 * t24 * t266 * t3;
        double t406 = -2640 * t22 * t24 * t266;
        double t407 = -3968 * psi * t22 * t24 * t266;
        double t408 = 4596 * t22 * t24 * t266 * t36;
        double t409 = 1140 * t153 * t22 * t24 * t266;
        double t410 = -8 * t159 * t22 * t24 * t266;
        double t411 = 3600 * t22 * t24 * t266 * t3;
        double t412 = -5760 * psi * t22 * t24 * t266 * t3;
        double t413 = 720 * t22 * t24 * t266 * t3 * t36;
        double t414 = 1440 * t153 * t22 * t24 * t266 * t3;
        double t415 = 3200 * psi * t24 * t266 * t73;
        double t416 = -1920 * t24 * t266 * t36 * t73;
        double t417 = -1280 * t153 * t24 * t266 * t73;
        double t418 = -3360 * t24 * t298;
        double t419 = 5248 * psi * t24 * t298;
        double t420 = -1176 * t24 * t298 * t36;
        double t421 = -760 * t153 * t24 * t298;
        double t422 = 48 * t159 * t24 * t298;
        double t423 = 2400 * t22 * t24 * t298;
        double t424 = -3840 * psi * t22 * t24 * t298;
        double t425 = 480 * t22 * t24 * t298 * t36;
        double t426 = 960 * t153 * t22 * t24 * t298;
        double t427 = -162 * t100;
        double t428 = -684 * psi * t100;
        double t429 = 336 * t100 * t36;
        double t430 = 346 * t100 * t153;
        double t431 = -4 * t100 * t159;
        double t432 = 702 * t100 * t3;
        double t433 = 1896 * psi * t100 * t3;
        double t434 = -1390 * t100 * t3 * t36;
        double t435 = -744 * t100 * t153 * t3;
        double t436 = 16 * t100 * t159 * t3;
        double t437 = -990 * t100 * t161;
        double t438 = -828 * psi * t100 * t161;
        double t439 = 1308 * t100 * t161 * t36;
        double t440 = 210 * t100 * t153 * t161;
        double t441 = -12 * t100 * t159 * t161;
        double t442 = 450 * t100 * t167;
        double t443 = -720 * psi * t100 * t167;
        double t444 = 90 * t100 * t167 * t36;
        double t445 = 180 * t100 * t153 * t167;
        double t446 = 756 * psi * t100 * t22;
        double t447 = -252 * t100 * t22 * t36;
        double t448 = -504 * t100 * t153 * t22;
        double t449 = -2520 * psi * t100 * t22 * t3;
        double t450 = 1176 * t100 * t22 * t3 * t36;
        double t451 = 1344 * t100 * t153 * t22 * t3;
        double t452 = 2100 * psi * t100 * t161 * t22;
        double t453 = -1260 * t100 * t161 * t22 * t36;
        double t454 = -840 * t100 * t153 * t161 * t22;
        double t455 = 1404 * t100 * t5;
        double t456 = 4080 * psi * t100 * t5;
        double t457 = -2948 * t100 * t36 * t5;
        double t458 = -1688 * t100 * t153 * t5;
        double t459 = 16 * t100 * t159 * t5;
        double t460 = -3960 * t100 * t3 * t5;
        double t461 = -3792 * psi * t100 * t3 * t5;
        double t462 = 5576 * t100 * t3 * t36 * t5;
        double t463 = 1056 * t100 * t153 * t3 * t5;
        double t464 = -32 * t100 * t159 * t3 * t5;
        double t465 = 2700 * t100 * t161 * t5;
        double t466 = -4320 * psi * t100 * t161 * t5;
        double t467 = 540 * t100 * t161 * t36 * t5;
        double t468 = 1080 * t100 * t153 * t161 * t5;
        double t469 = -5040 * psi * t100 * t22 * t5;
        double t470 = 2352 * t100 * t22 * t36 * t5;
        double t471 = 2688 * t100 * t153 * t22 * t5;
        double t472 = 8400 * psi * t100 * t22 * t3 * t5;
        double t473 = -5040 * t100 * t22 * t3 * t36 * t5;
        double t474 = -3360 * t100 * t153 * t22 * t3 * t5;
        double t475 = -3960 * t100 * t266;
        double t476 = -4272 * psi * t100 * t266;
        double t477 = 5920 * t100 * t266 * t36;
        double t478 = 1272 * t100 * t153 * t266;
        double t479 = -16 * t100 * t159 * t266;
        double t480 = 5400 * t100 * t266 * t3;
        double t481 = -8640 * psi * t100 * t266 * t3;
        double t482 = 1080 * t100 * t266 * t3 * t36;
        double t484 = 8400 * psi * t100 * t22 * t266;
        double t485 = -5040 * t100 * t22 * t266 * t36;
        double t486 = -3360 * t100 * t153 * t22 * t266;
        double t487 = 3600 * t100 * t298;
        double t488 = -5760 * psi * t100 * t298;
        double t489 = 720 * t100 * t298 * t36;
        double t490 = 1440 * t100 * t153 * t298;
        double t492 = 648 * psi * t491;
        double t493 = -216 * t36 * t491;
        double t495 = -2160 * psi * t3 * t491;
        double t496 = 1008 * t3 * t36 * t491;
        double t497 = 1152 * t153 * t3 * t491;
        double t498 = 1800 * psi * t161 * t491;
        double t499 = -1080 * t161 * t36 * t491;
        double t500 = -720 * t153 * t161 * t491;
        double t501 = -4320 * psi * t491 * t5;
        double t502 = 2016 * t36 * t491 * t5;
        double t503 = 2304 * t153 * t491 * t5;
        double t504 = 7200 * psi * t3 * t491 * t5;
        double t505 = -4320 * t3 * t36 * t491 * t5;
        double t506 = -2880 * t153 * t3 * t491 * t5;
        double t507 = 7200 * psi * t266 * t491;
        double t509 = -2880 * t153 * t266 * t491;
        double t510 =
            -32 + t151 + t152 + t154 + t155 + t156 + t157 + t158 + t160 + t162 + t163 +
            t164 + t165 + t166 + t168 + t169 + t170 + t171 + t172 + t173 + t174 + t175 +
            t176 + t177 + t178 + t179 + t180 + t181 + t182 + t183 + t184 + t185 + t186 +
            t187 + t188 + t189 + t190 + t191 + t192 + t193 + t194 + t195 + t196 + t197 +
            t198 + t199 + t200 + t201 + t202 + t203 + t204 + t205 + t206 + t207 + t209 +
            t210 + t211 + t212 + t213 + t214 + t215 + t216 + t217 + t218 + t219 + t220 +
            t221 + t222 + t223 + t224 + t225 + t226 + t227 + t228 + t229 + t230 + t231 +
            t232 + t233 + t234 + t235 + t236 + t237 + t238 + t239 + t240 + t241 + t242 +
            t243 + t244 + t245 + t246 + t247 + t248 + t249 + t250 + t251 + t252 + t253 +
            t254 + t255 + t256 + t257 + t258 + t259 + t260 + t261 + t262 + t263 + t264 +
            t265 + t267 + t268 + t269 + t270 + t271 + t272 + t273 + t274 + t275 + t276 +
            t277 + t278 + t279 + t280 + t281 + t282 + t283 + t284 + t285 + t286 + t287 +
            t288 + t289 + t290 + t291 + t292 + t293 + t294 + t295 + t296 + t297 + t299 +
            t300 + t301 + t302 + t303 + t304 + t305 + t306 + t307 + t308 + t309 + t310 +
            t311 + t312 + t313 + t314 + t315 + t316 + t317 + t318 + t319 + t320 + t321 +
            t322 + t323 + t324 + t325 + t326 + t327 + t328 + t329 + t330 + t331 + t332 +
            t333 + t334 + t335 + t336 + t337 + t338 + t339 + t340 + t341 + t342 + t343 +
            t344 + t345 + t346 + t347 + t348 + t349 + t350 + t351 + t352 + t353 + t354 +
            t355 + t356 + t357 + t358 + t359 + t360 + t361 + t362 + t363 + t364 + t365 +
            t366 + t367 + t368 + t369 + t370 + t371 + t372 + t373 + t374 + t375 + t376 +
            t377 + t378 + t379 + t380 + t381 + t382 + t383 + t384 + t385 + t386 + t387 +
            t388 + t389 + t390 + t391 + t392 + t393 + t394 + t395 + t396 + t397 + t398 +
            t399 + t400 + t401 + t402 + t403 + t404 + t405 + t406 + t407 + t408 + t409 +
            t410 + t411 + t412 + t413 + t414 + t415 + t416 + t417 + t418 + t419 + t420 +
            t421 + t422 + t423 + t424 + t425 + t426 + t427 + t428 + t429 + t430 + t431 +
            t432 + t433 + t434 + t435 + t436 + t437 + t438 + t439 + t440 + t441 + t442 +
            t443 + t444 + t445 + t446 + t447 + t448 + t449 + t450 + t451 + t452 + t453 +
            t454 + t455 + t456 + t457 + t458 + t459 + t460 + t461 + t462 + t463 + t464 +
            t465 + t466 + t467 + t468 + t469 + t470 + t471 + t472 + t473 + t474 + t475 +
            t476 + t477 + t478 + t479 + t480 + t481 + t482 + t483 + t484 + t485 + t486 +
            t487 + t488 + t489 + t490 + t492 + t493 + t494 + t495 + t496 + t497 + t498 +
            t499 + t500 + t501 + t502 + t503 + t504 + t505 + t506 + t507 + t508 + t509;
        double t511 = std::pow(t510, -1);
        double t522 = 36 * t167;
        double t67 = 14 * t22;
        double t74 = -6 * t73;
        double t551 = -40 * psi * t167 * t73;
        double t86 = 34 * t24;
        double t91 = -30 * t22 * t24;
        double t677 = 150 * t167 * t22 * t24;
        double t718 = 360 * t153 * t161 * t22 * t24 * t5;
        double t101 = -36 * t100;
        double t512 = -21 * psi;
        double t513 = 3 * t36;
        double t514 = 84 * t3;
        double t515 = 32 * psi * t3;
        double t516 = 6 * t3 * t36;
        double t517 = -2 * t153 * t3;
        double t518 = -96 * t161;
        double t519 = 25 * psi * t161;
        double t520 = -15 * t161 * t36;
        double t521 = 8 * t153 * t161;
        double t523 = -36 * psi * t167;
        double t524 = 6 * t167 * t36;
        double t525 = -6 * t153 * t167;
        double t526 = 34 * t22;
        double t527 = 65 * psi * t22;
        double t528 = -29 * t22 * t36;
        double t529 = -122 * t22 * t3;
        double t530 = -120 * psi * t22 * t3;
        double t531 = 40 * t22 * t3 * t36;
        double t532 = 2 * t153 * t22 * t3;
        double t533 = 142 * t161 * t22;
        double t534 = -21 * psi * t161 * t22;
        double t535 = 17 * t161 * t22 * t36;
        double t536 = -8 * t153 * t161 * t22;
        double t537 = -54 * t167 * t22;
        double t538 = 76 * psi * t167 * t22;
        double t539 = -28 * t167 * t22 * t36;
        double t540 = 6 * t153 * t167 * t22;
        double t541 = -12 * t73;
        double t542 = -68 * psi * t73;
        double t543 = 52 * t36 * t73;
        double t544 = 44 * t3 * t73;
        double t545 = 152 * psi * t3 * t73;
        double t546 = -116 * t3 * t36 * t73;
        double t547 = -52 * t161 * t73;
        double t548 = -44 * psi * t161 * t73;
        double t549 = 44 * t161 * t36 * t73;
        double t550 = 20 * t167 * t73;
        double t552 = 20 * t167 * t36 * t73;
        double t553 = 24 * psi * t208;
        double t554 = -24 * t208 * t36;
        double t555 = -64 * psi * t208 * t3;
        double t556 = 64 * t208 * t3 * t36;
        double t557 = 40 * psi * t161 * t208;
        double t558 = -40 * t161 * t208 * t36;
        double t559 = 228 * t5;
        double t560 = 49 * psi * t5;
        double t561 = -54 * t36 * t5;
        double t562 = -(t153 * t5);
        double t563 = -528 * t3 * t5;
        double t564 = 228 * psi * t3 * t5;
        double t565 = -72 * t3 * t36 * t5;
        double t566 = 60 * t153 * t3 * t5;
        double t567 = 24 * t159 * t3 * t5;
        double t568 = 300 * t161 * t5;
        double t569 = -377 * psi * t161 * t5;
        double t570 = 162 * t161 * t36 * t5;
        double t571 = -67 * t153 * t161 * t5;
        double t572 = -18 * t159 * t161 * t5;
        double t573 = -320 * t22 * t5;
        double t574 = -309 * psi * t22 * t5;
        double t575 = 314 * t22 * t36 * t5;
        double t576 = -51 * t153 * t22 * t5;
        double t577 = 760 * t22 * t3 * t5;
        double t578 = -164 * psi * t22 * t3 * t5;
        double t579 = -72 * t22 * t3 * t36 * t5;
        double t580 = -20 * t153 * t22 * t3 * t5;
        double t581 = -24 * t159 * t22 * t3 * t5;
        double t582 = -440 * t161 * t22 * t5;
        double t583 = 733 * psi * t161 * t22 * t5;
        double t584 = -422 * t161 * t22 * t36 * t5;
        double t585 = 111 * t153 * t161 * t22 * t5;
        double t586 = 18 * t159 * t161 * t22 * t5;
        double t587 = 112 * t5 * t73;
        double t588 = 432 * psi * t5 * t73;
        double t589 = -504 * t36 * t5 * t73;
        double t590 = 104 * t153 * t5 * t73;
        double t591 = -272 * t3 * t5 * t73;
        double t592 = -296 * psi * t3 * t5 * t73;
        double t593 = 504 * t3 * t36 * t5 * t73;
        double t594 = -128 * t153 * t3 * t5 * t73;
        double t595 = 160 * t161 * t5 * t73;
        double t596 = -360 * psi * t161 * t5 * t73;
        double t597 = 240 * t161 * t36 * t5 * t73;
        double t598 = -40 * t153 * t161 * t5 * t73;
        double t599 = -176 * psi * t208 * t5;
        double t600 = 224 * t208 * t36 * t5;
        double t601 = -48 * t153 * t208 * t5;
        double t602 = 240 * psi * t208 * t3 * t5;
        double t603 = -320 * t208 * t3 * t36 * t5;
        double t604 = 80 * t153 * t208 * t3 * t5;
        double t605 = -696 * t266;
        double t606 = 416 * psi * t266;
        double t607 = -32 * t266 * t36;
        double t608 = 28 * t153 * t266;
        double t609 = 20 * t159 * t266;
        double t610 = 792 * t266 * t3;
        double t611 = -1184 * psi * t266 * t3;
        double t612 = 612 * t266 * t3 * t36;
        double t613 = -184 * t153 * t266 * t3;
        double t614 = -36 * t159 * t266 * t3;
        double t615 = 968 * t22 * t266;
        double t616 = -264 * psi * t22 * t266;
        double t617 = -352 * t22 * t266 * t36;
        double t618 = 108 * t153 * t22 * t266;
        double t619 = -20 * t159 * t22 * t266;
        double t620 = -1128 * t22 * t266 * t3;
        double t621 = 2120 * psi * t22 * t266 * t3;
        double t622 = -1396 * t22 * t266 * t3 * t36;
        double t623 = 368 * t153 * t22 * t266 * t3;
        double t624 = 36 * t159 * t22 * t266 * t3;
        double t625 = -336 * t266 * t73;
        double t626 = -448 * psi * t266 * t73;
        double t627 = 928 * t266 * t36 * t73;
        double t628 = -320 * t153 * t266 * t73;
        double t629 = 400 * t266 * t3 * t73;
        double t630 = 720 * t266 * t3 * t36 * t73;
        double t631 = -160 * t153 * t266 * t3 * t73;
        double t632 = 320 * psi * t208 * t266;
        double t633 = -480 * t208 * t266 * t36;
        double t634 = 160 * t153 * t208 * t266;
        double t635 = 672 * t298;
        double t636 = -1148 * psi * t298;
        double t637 = 640 * t298 * t36;
        double t638 = -148 * t153 * t298;
        double t639 = -16 * t159 * t298;
        double t640 = -928 * t22 * t298;
        double t641 = 1916 * psi * t22 * t298;
        double t642 = -1344 * t22 * t298 * t36;
        double t643 = 340 * t153 * t22 * t298;
        double t644 = 16 * t159 * t22 * t298;
        double t645 = 320 * t298 * t73;
        double t646 = -800 * psi * t298 * t73;
        double t647 = 640 * t298 * t36 * t73;
        double t648 = -160 * t153 * t298 * t73;
        double t649 = 110 * t24;
        double t650 = 190 * psi * t24;
        double t651 = -86 * t24 * t36;
        double t652 = -46 * t153 * t24;
        double t653 = -410 * t24 * t3;
        double t654 = -400 * psi * t24 * t3;
        double t655 = 230 * t24 * t3 * t36;
        double t656 = 100 * t153 * t24 * t3;
        double t657 = 498 * t161 * t24;
        double t658 = 34 * psi * t161 * t24;
        double t659 = -242 * t161 * t24 * t36;
        double t660 = 22 * t153 * t161 * t24;
        double t661 = -198 * t167 * t24;
        double t662 = 192 * psi * t167 * t24;
        double t663 = 114 * t167 * t24 * t36;
        double t664 = -108 * t153 * t167 * t24;
        double t665 = -78 * t22 * t24;
        double t666 = -392 * psi * t22 * t24;
        double t667 = 200 * t22 * t24 * t36;
        double t668 = 130 * t153 * t22 * t24;
        double t669 = 298 * t22 * t24 * t3;
        double t670 = 952 * psi * t22 * t24 * t3;
        double t671 = -470 * t22 * t24 * t3 * t36;
        double t672 = -380 * t153 * t22 * t24 * t3;
        double t673 = -370 * t161 * t22 * t24;
        double t674 = -392 * psi * t161 * t22 * t24;
        double t675 = 288 * t161 * t22 * t24 * t36;
        double t676 = 214 * t153 * t161 * t22 * t24;
        double t678 = -200 * psi * t167 * t22 * t24;
        double t679 = -50 * t167 * t22 * t24 * t36;
        double t680 = 100 * t153 * t167 * t22 * t24;
        double t681 = 204 * psi * t24 * t73;
        double t682 = -132 * t24 * t36 * t73;
        double t683 = -72 * t153 * t24 * t73;
        double t684 = -568 * psi * t24 * t3 * t73;
        double t685 = 328 * t24 * t3 * t36 * t73;
        double t686 = 240 * t153 * t24 * t3 * t73;
        double t687 = 380 * psi * t161 * t24 * t73;
        double t688 = -180 * t161 * t24 * t36 * t73;
        double t689 = -200 * t153 * t161 * t24 * t73;
        double t690 = -1020 * t24 * t5;
        double t691 = -918 * psi * t24 * t5;
        double t692 = 980 * t24 * t36 * t5;
        double t693 = 78 * t153 * t24 * t5;
        double t694 = 16 * t159 * t24 * t5;
        double t695 = 2504 * t24 * t3 * t5;
        double t696 = -168 * psi * t24 * t3 * t5;
        double t697 = -1236 * t24 * t3 * t36 * t5;
        double t698 = 112 * t153 * t24 * t3 * t5;
        double t699 = -60 * t159 * t24 * t3 * t5;
        double t700 = -1500 * t161 * t24 * t5;
        double t701 = 1894 * psi * t161 * t24 * t5;
        double t702 = 40 * t161 * t24 * t36 * t5;
        double t703 = -470 * t153 * t161 * t24 * t5;
        double t704 = 36 * t159 * t161 * t24 * t5;
        double t705 = 716 * t22 * t24 * t5;
        double t706 = 2464 * psi * t22 * t24 * t5;
        double t707 = -2192 * t22 * t24 * t36 * t5;
        double t708 = -264 * t153 * t22 * t24 * t5;
        double t709 = -4 * t159 * t22 * t24 * t5;
        double t710 = -1800 * t22 * t24 * t3 * t5;
        double t711 = -2032 * psi * t22 * t24 * t3 * t5;
        double t712 = 2596 * t22 * t24 * t3 * t36 * t5;
        double t713 = 264 * t153 * t22 * t24 * t3 * t5;
        double t714 = 12 * t159 * t22 * t24 * t3 * t5;
        double t715 = 1100 * t161 * t22 * t24 * t5;
        double t716 = -1840 * psi * t161 * t22 * t24 * t5;
        double t717 = 380 * t161 * t22 * t24 * t36 * t5;
        double t719 = -1472 * psi * t24 * t5 * t73;
        double t720 = 1376 * t24 * t36 * t5 * t73;
        double t721 = 96 * t153 * t24 * t5 * t73;
        double t722 = 2080 * psi * t24 * t3 * t5 * t73;
        double t723 = -160 * t153 * t24 * t3 * t5 * t73;
        double t724 = 3048 * t24 * t266;
        double t725 = -528 * psi * t24 * t266;
        double t726 = -1800 * t24 * t266 * t36;
        double t727 = 384 * t153 * t24 * t266;
        double t728 = -48 * t159 * t24 * t266;
        double t729 = -3656 * t24 * t266 * t3;
        double t730 = 5488 * psi * t24 * t266 * t3;
        double t731 = -1328 * t24 * t266 * t3 * t36;
        double t732 = -576 * t153 * t24 * t266 * t3;
        double t733 = 72 * t159 * t24 * t266 * t3;
        double t734 = -2120 * t22 * t24 * t266;
        double t735 = -2624 * psi * t22 * t24 * t266;
        double t736 = 4472 * t22 * t24 * t266 * t36;
        double t737 = -616 * t153 * t22 * t24 * t266;
        double t738 = 8 * t159 * t22 * t24 * t266;
        double t739 = 2600 * t22 * t24 * t266 * t3;
        double t740 = -4960 * psi * t22 * t24 * t266 * t3;
        double t741 = 2120 * t22 * t24 * t266 * t3 * t36;
        double t742 = 240 * t153 * t22 * t24 * t266 * t3;
        double t743 = 2640 * psi * t24 * t266 * t73;
        double t744 = -3120 * t24 * t266 * t36 * t73;
        double t745 = 480 * t153 * t24 * t266 * t73;
        double t746 = -2896 * t24 * t298;
        double t747 = 4936 * psi * t24 * t298;
        double t748 = -1936 * t24 * t298 * t36;
        double t749 = -136 * t153 * t24 * t298;
        double t750 = 32 * t159 * t24 * t298;
        double t751 = 2000 * t22 * t24 * t298;
        double t752 = -4160 * psi * t22 * t24 * t298;
        double t753 = 2320 * t22 * t24 * t298 * t36;
        double t754 = -160 * t153 * t22 * t24 * t298;
        double t755 = -126 * t100;
        double t756 = -548 * psi * t100;
        double t757 = 210 * t100 * t36;
        double t758 = 296 * t100 * t153;
        double t759 = 498 * t100 * t3;
        double t760 = 1416 * psi * t100 * t3;
        double t761 = -590 * t100 * t3 * t36;
        double t762 = -844 * t100 * t153 * t3;
        double t763 = -642 * t100 * t161;
        double t764 = -708 * psi * t100 * t161;
        double t765 = 630 * t100 * t161 * t36;
        double t766 = 408 * t100 * t153 * t161;
        double t767 = 270 * t100 * t167;
        double t768 = -240 * psi * t100 * t167;
        double t769 = -330 * t100 * t167 * t36;
        double t770 = 300 * t100 * t153 * t167;
        double t771 = 564 * psi * t100 * t22;
        double t772 = -204 * t100 * t22 * t36;
        double t773 = -360 * t100 * t153 * t22;
        double t774 = -1624 * psi * t100 * t22 * t3;
        double t775 = 424 * t100 * t22 * t3 * t36;
        double t776 = 1200 * t100 * t153 * t22 * t3;
        double t777 = 1140 * psi * t100 * t161 * t22;
        double t778 = -140 * t100 * t161 * t22 * t36;
        double t779 = -1000 * t100 * t153 * t161 * t22;
        double t780 = 1140 * t100 * t5;
        double t781 = 3416 * psi * t100 * t5;
        double t782 = -2596 * t100 * t36 * t5;
        double t783 = -1088 * t100 * t153 * t5;
        double t784 = -8 * t100 * t159 * t5;
        double t785 = -2952 * t100 * t3 * t5;
        double t786 = -3216 * psi * t100 * t3 * t5;
        double t787 = 4048 * t100 * t3 * t36 * t5;
        double t788 = 944 * t100 * t153 * t3 * t5;
        double t789 = 24 * t100 * t159 * t3 * t5;
        double t790 = 1860 * t100 * t161 * t5;
        double t791 = -2280 * psi * t100 * t161 * t5;
        double t792 = -1020 * t100 * t161 * t36 * t5;
        double t793 = 1440 * t100 * t153 * t161 * t5;
        double t794 = -4016 * psi * t100 * t22 * t5;
        double t795 = 2624 * t100 * t22 * t36 * t5;
        double t796 = 1392 * t100 * t153 * t22 * t5;
        double t797 = 5840 * psi * t100 * t22 * t3 * t5;
        double t798 = -3520 * t100 * t22 * t3 * t36 * t5;
        double t799 = -2320 * t100 * t153 * t22 * t3 * t5;
        double t800 = -3336 * t100 * t266;
        double t801 = -3728 * psi * t100 * t266;
        double t802 = 6056 * t100 * t266 * t36;
        double t803 = -64 * t100 * t153 * t266;
        double t804 = 16 * t100 * t159 * t266;
        double t805 = 4200 * t100 * t266 * t3;
        double t806 = -6240 * psi * t100 * t266 * t3;
        double t807 = -120 * t100 * t266 * t3 * t36;
        double t808 = 7120 * psi * t100 * t22 * t266;
        double t809 = -6480 * t100 * t22 * t266 * t36;
        double t810 = -640 * t100 * t153 * t22 * t266;
        double t811 = 3120 * t100 * t298;
        double t812 = -5280 * psi * t100 * t298;
        double t813 = 1200 * t100 * t298 * t36;
        double t814 = 960 * t100 * t153 * t298;
        double t815 = 504 * psi * t491;
        double t816 = -72 * t36 * t491;
        double t817 = -1488 * psi * t3 * t491;
        double t818 = 48 * t3 * t36 * t491;
        double t819 = 1440 * t153 * t3 * t491;
        double t820 = 1080 * psi * t161 * t491;
        double t821 = 120 * t161 * t36 * t491;
        double t822 = -1200 * t153 * t161 * t491;
        double t823 = -3552 * psi * t491 * t5;
        double t824 = 1536 * t36 * t491 * t5;
        double t825 = 2016 * t153 * t491 * t5;
        double t826 = 5280 * psi * t3 * t491 * t5;
        double t827 = -1920 * t3 * t36 * t491 * t5;
        double t828 = -3360 * t153 * t3 * t491 * t5;
        double t829 = 6240 * psi * t266 * t491;
        double t830 = -1920 * t153 * t266 * t491;
        double t831 =
            -24 + t222 + t237 + t292 + t394 + t483 + t494 + t508 + t512 + t513 + t514 +
            t515 + t516 + t517 + t518 + t519 + t520 + t521 + t522 + t523 + t524 + t525 +
            t526 + t527 + t528 + t529 + t530 + t531 + t532 + t533 + t534 + t535 + t536 +
            t537 + t538 + t539 + t540 + t541 + t542 + t543 + t544 + t545 + t546 + t547 +
            t548 + t549 + t550 + t551 + t552 + t553 + t554 + t555 + t556 + t557 + t558 +
            t559 + t560 + t561 + t562 + t563 + t564 + t565 + t566 + t567 + t568 + t569 +
            t570 + t571 + t572 + t573 + t574 + t575 + t576 + t577 + t578 + t579 + t580 +
            t581 + t582 + t583 + t584 + t585 + t586 + t587 + t588 + t589 + t590 + t591 +
            t592 + t593 + t594 + t595 + t596 + t597 + t598 + t599 + t600 + t601 + t602 +
            t603 + t604 + t605 + t606 + t607 + t608 + t609 + t610 + t611 + t612 + t613 +
            t614 + t615 + t616 + t617 + t618 + t619 + t620 + t621 + t622 + t623 + t624 +
            t625 + t626 + t627 + t628 + t629 + t630 + t631 + t632 + t633 + t634 + t635 +
            t636 + t637 + t638 + t639 + t640 + t641 + t642 + t643 + t644 + t645 + t646 +
            t647 + t648 + t649 + t650 + t651 + t652 + t653 + t654 + t655 + t656 + t657 +
            t658 + t659 + t660 + t661 + t662 + t663 + t664 + t665 + t666 + t667 + t668 +
            t669 + t670 + t671 + t672 + t673 + t674 + t675 + t676 + t677 + t678 + t679 +
            t680 + t681 + t682 + t683 + t684 + t685 + t686 + t687 + t688 + t689 + t690 +
            t691 + t692 + t693 + t694 + t695 + t696 + t697 + t698 + t699 + t700 + t701 +
            t702 + t703 + t704 + t705 + t706 + t707 + t708 + t709 + t710 + t711 + t712 +
            t713 + t714 + t715 + t716 + t717 + t718 + t719 + t720 + t721 + t722 + t723 +
            t724 + t725 + t726 + t727 + t728 + t729 + t730 + t731 + t732 + t733 + t734 +
            t735 + t736 + t737 + t738 + t739 + t740 + t741 + t742 + t743 + t744 + t745 +
            t746 + t747 + t748 + t749 + t750 + t751 + t752 + t753 + t754 + t755 + t756 +
            t757 + t758 + t759 + t760 + t761 + t762 + t763 + t764 + t765 + t766 + t767 +
            t768 + t769 + t770 + t771 + t772 + t773 + t774 + t775 + t776 + t777 + t778 +
            t779 + t780 + t781 + t782 + t783 + t784 + t785 + t786 + t787 + t788 + t789 +
            t790 + t791 + t792 + t793 + t794 + t795 + t796 + t797 + t798 + t799 + t800 +
            t801 + t802 + t803 + t804 + t805 + t806 + t807 + t808 + t809 + t810 + t811 +
            t812 + t813 + t814 + t815 + t816 + t817 + t818 + t819 + t820 + t821 + t822 +
            t823 + t824 + t825 + t826 + t827 + t828 + t829 + t830;
        double t833 = -6 * psi;
        double t834 = 8 * t36;
        double t835 = 44 * t3;
        double t836 = 4 * psi * t3;
        double t837 = -37 * t3 * t36;
        double t838 = -9 * t153 * t3;
        double t839 = -72 * t161;
        double t840 = 46 * psi * t161;
        double t841 = 12 * t161 * t36;
        double t842 = 6 * t153 * t161;
        double t843 = -60 * psi * t167;
        double t844 = 33 * t167 * t36;
        double t845 = -3 * t153 * t167;
        double t846 = 37 * psi * t22;
        double t847 = -19 * t22 * t36;
        double t848 = -78 * t22 * t3;
        double t849 = -119 * psi * t22 * t3;
        double t850 = 140 * t22 * t3 * t36;
        double t851 = 55 * t153 * t22 * t3;
        double t852 = 130 * t161 * t22;
        double t853 = 35 * psi * t161 * t22;
        double t854 = -153 * t161 * t22 * t36;
        double t855 = -4 * t153 * t161 * t22;
        double t856 = -66 * t167 * t22;
        double t857 = 95 * psi * t167 * t22;
        double t858 = -16 * t167 * t22 * t36;
        double t859 = -19 * t153 * t167 * t22;
        double t860 = -40 * psi * t73;
        double t861 = -6 * t36 * t73;
        double t862 = 34 * t3 * t73;
        double t863 = 160 * psi * t3 * t73;
        double t864 = -78 * t3 * t36 * t73;
        double t865 = -58 * t161 * t73;
        double t866 = -128 * psi * t161 * t73;
        double t867 = 142 * t161 * t36 * t73;
        double t868 = 30 * t167 * t73;
        double t869 = -10 * t167 * t36 * t73;
        double t870 = 12 * psi * t208;
        double t871 = 12 * t208 * t36;
        double t872 = -56 * psi * t208 * t3;
        double t873 = -8 * t208 * t3 * t36;
        double t874 = 60 * psi * t161 * t208;
        double t875 = -20 * t161 * t208 * t36;
        double t876 = 60 * t5;
        double t877 = 26 * psi * t5;
        double t878 = -50 * t36 * t5;
        double t879 = -36 * t153 * t5;
        double t880 = -224 * t3 * t5;
        double t881 = 112 * psi * t3 * t5;
        double t882 = 114 * t3 * t36 * t5;
        double t883 = -10 * t153 * t3 * t5;
        double t884 = 8 * t159 * t3 * t5;
        double t885 = 180 * t161 * t5;
        double t886 = -290 * psi * t161 * t5;
        double t887 = 96 * t161 * t36 * t5;
        double t888 = 30 * t153 * t161 * t5;
        double t889 = -16 * t159 * t161 * t5;
        double t890 = -104 * t22 * t5;
        double t891 = -230 * psi * t22 * t5;
        double t892 = 126 * t22 * t36 * t5;
        double t893 = 208 * t153 * t22 * t5;
        double t894 = 392 * t22 * t3 * t5;
        double t895 = 268 * psi * t22 * t3 * t5;
        double t896 = -606 * t22 * t3 * t36 * t5;
        double t897 = -46 * t153 * t22 * t3 * t5;
        double t898 = -8 * t159 * t22 * t3 * t5;
        double t899 = -320 * t161 * t22 * t5;
        double t900 = 370 * psi * t161 * t22 * t5;
        double t901 = 144 * t161 * t22 * t36 * t5;
        double t902 = -210 * t153 * t161 * t22 * t5;
        double t903 = 16 * t159 * t161 * t22 * t5;
        double t904 = 44 * t5 * t73;
        double t905 = 240 * psi * t5 * t73;
        double t906 = 84 * t36 * t5 * t73;
        double t907 = -368 * t153 * t5 * t73;
        double t908 = -168 * t3 * t5 * t73;
        double t909 = -472 * psi * t3 * t5 * t73;
        double t910 = 304 * t3 * t36 * t5 * t73;
        double t911 = 336 * t153 * t3 * t5 * t73;
        double t912 = 140 * t161 * t5 * t73;
        double t913 = -120 * psi * t161 * t5 * t73;
        double t914 = -180 * t161 * t36 * t5 * t73;
        double t915 = 160 * t153 * t161 * t5 * t73;
        double t916 = -64 * psi * t208 * t5;
        double t917 = -112 * t208 * t36 * t5;
        double t918 = 176 * t153 * t208 * t5;
        double t919 = 160 * psi * t208 * t3 * t5;
        double t920 = 80 * t208 * t3 * t36 * t5;
        double t921 = -240 * t153 * t208 * t3 * t5;
        double t922 = -144 * t266;
        double t923 = -8 * psi * t266;
        double t924 = 144 * t266 * t36;
        double t925 = 4 * t153 * t266;
        double t926 = 4 * t159 * t266;
        double t927 = 272 * t266 * t3;
        double t928 = -360 * psi * t266 * t3;
        double t929 = -56 * t266 * t3 * t36;
        double t930 = 164 * t153 * t266 * t3;
        double t931 = -20 * t159 * t266 * t3;
        double t932 = 248 * t22 * t266;
        double t933 = 412 * psi * t22 * t266;
        double t934 = -476 * t22 * t266 * t36;
        double t935 = -180 * t153 * t22 * t266;
        double t936 = -4 * t159 * t22 * t266;
        double t937 = -472 * t22 * t266 * t3;
        double t938 = 260 * psi * t22 * t266 * t3;
        double t939 = 820 * t22 * t266 * t3 * t36;
        double t940 = -628 * t153 * t22 * t266 * t3;
        double t941 = 20 * t159 * t22 * t266 * t3;
        double t942 = -104 * t266 * t73;
        double t943 = -400 * psi * t266 * t73;
        double t944 = -56 * t266 * t36 * t73;
        double t945 = 560 * t153 * t266 * t73;
        double t946 = 200 * t266 * t3 * t73;
        double t947 = -600 * t266 * t3 * t36 * t73;
        double t948 = 400 * t153 * t266 * t3 * t73;
        double t949 = 80 * psi * t208 * t266;
        double t950 = 240 * t208 * t266 * t36;
        double t951 = -320 * t153 * t208 * t266;
        double t952 = 112 * t298;
        double t953 = -40 * psi * t298;
        double t954 = -248 * t298 * t36;
        double t955 = 184 * t153 * t298;
        double t956 = -8 * t159 * t298;
        double t957 = -192 * t22 * t298;
        double t958 = -200 * psi * t22 * t298;
        double t959 = 952 * t22 * t298 * t36;
        double t960 = -568 * t153 * t22 * t298;
        double t961 = 8 * t159 * t22 * t298;
        double t962 = 80 * t298 * t73;
        double t963 = 160 * psi * t298 * t73;
        double t964 = -560 * t298 * t36 * t73;
        double t965 = 320 * t153 * t298 * t73;
        double t966 = 45 * psi * t24;
        double t967 = -46 * t24 * t36;
        double t968 = -35 * t153 * t24;
        double t969 = -190 * t24 * t3;
        double t970 = -87 * psi * t24 * t3;
        double t971 = 255 * t24 * t3 * t36;
        double t972 = 26 * t153 * t24 * t3;
        double t973 = 318 * t161 * t24;
        double t974 = -197 * psi * t161 * t24;
        double t975 = -108 * t161 * t24 * t36;
        double t976 = -3 * t153 * t161 * t24;
        double t977 = -162 * t167 * t24;
        double t978 = 351 * psi * t167 * t24;
        double t979 = -249 * t167 * t24 * t36;
        double t980 = 48 * t153 * t167 * t24;
        double t981 = -160 * psi * t22 * t24;
        double t982 = 51 * t22 * t24 * t36;
        double t983 = 141 * t153 * t22 * t24;
        double t984 = 170 * t22 * t24 * t3;
        double t985 = 616 * psi * t22 * t24 * t3;
        double t986 = -580 * t22 * t24 * t3 * t36;
        double t987 = -214 * t153 * t22 * t24 * t3;
        double t988 = -290 * t161 * t22 * t24;
        double t989 = -400 * psi * t161 * t22 * t24;
        double t990 = 711 * t161 * t22 * t24 * t36;
        double t991 = -15 * t153 * t161 * t22 * t24;
        double t992 = -280 * psi * t167 * t22 * t24;
        double t993 = 110 * t167 * t22 * t24 * t36;
        double t994 = 20 * t153 * t167 * t22 * t24;
        double t995 = 84 * psi * t24 * t73;
        double t996 = 36 * t24 * t36 * t73;
        double t997 = -120 * t153 * t24 * t73;
        double t998 = -392 * psi * t24 * t3 * t73;
        double t999 = 120 * t24 * t3 * t36 * t73;
        double t1000 = 272 * t153 * t24 * t3 * t73;
        double t1001 = 420 * psi * t161 * t24 * t73;
        double t1002 = -300 * t161 * t24 * t36 * t73;
        double t1003 = -120 * t153 * t161 * t24 * t73;
        double t1004 = -252 * t24 * t5;
        double t1005 = -198 * psi * t24 * t5;
        double t1006 = 174 * t24 * t36 * t5;
        double t1007 = 288 * t153 * t24 * t5;
        double t1008 = -12 * t159 * t24 * t5;
        double t1009 = 952 * t24 * t3 * t5;
        double t1010 = -388 * psi * t24 * t3 * t5;
        double t1011 = -476 * t24 * t3 * t36 * t5;
        double t1012 = -100 * t153 * t24 * t3 * t5;
        double t1013 = 12 * t159 * t24 * t3 * t5;
        double t1014 = -780 * t161 * t24 * t5;
        double t1015 = 1570 * psi * t161 * t24 * t5;
        double t1016 = -882 * t161 * t24 * t36 * t5;
        double t1017 = 60 * t153 * t161 * t24 * t5;
        double t1018 = 32 * t159 * t161 * t24 * t5;
        double t1019 = 220 * t22 * t24 * t5;
        double t1020 = 912 * psi * t22 * t24 * t5;
        double t1021 = -56 * t22 * t24 * t36 * t5;
        double t1022 = -1088 * t153 * t22 * t24 * t5;
        double t1023 = 12 * t159 * t22 * t24 * t5;
        double t1024 = -840 * t22 * t24 * t3 * t5;
        double t1025 = -1536 * psi * t22 * t24 * t3 * t5;
        double t1026 = 1700 * t22 * t24 * t3 * t36 * t5;
        double t1027 = 704 * t153 * t22 * t24 * t3 * t5;
        double t1028 = -28 * t159 * t22 * t24 * t3 * t5;
        double t1029 = 700 * t161 * t22 * t24 * t5;
        double t1030 = -1040 * psi * t161 * t22 * t24 * t5;
        double t1031 = -20 * t161 * t22 * t24 * t36 * t5;
        double t1032 = -448 * psi * t24 * t5 * t73;
        double t1033 = -480 * t24 * t36 * t5 * t73;
        double t1034 = 928 * t153 * t24 * t5 * t73;
        double t1035 = 1120 * psi * t24 * t3 * t5 * t73;
        double t1036 = -1120 * t153 * t24 * t3 * t5 * t73;
        double t1037 = 600 * t24 * t266;
        double t1038 = 60 * psi * t24 * t266;
        double t1039 = -256 * t24 * t266 * t36;
        double t1040 = -412 * t153 * t24 * t266;
        double t1041 = 8 * t159 * t24 * t266;
        double t1042 = -1144 * t24 * t266 * t3;
        double t1043 = 1892 * psi * t24 * t266 * t3;
        double t1044 = -404 * t24 * t266 * t3 * t36;
        double t1045 = -384 * t153 * t24 * t266 * t3;
        double t1046 = 40 * t159 * t24 * t266 * t3;
        double t1047 = -520 * t22 * t24 * t266;
        double t1048 = -1344 * psi * t22 * t24 * t266;
        double t1049 = 124 * t22 * t24 * t266 * t36;
        double t1050 = 1756 * t153 * t22 * t24 * t266;
        double t1051 = -16 * t159 * t22 * t24 * t266;
        double t1052 = 1000 * t22 * t24 * t266 * t3;
        double t1053 = -800 * psi * t22 * t24 * t266 * t3;
        double t1054 = -1400 * t22 * t24 * t266 * t3 * t36;
        double t1055 = 1200 * t153 * t22 * t24 * t266 * t3;
        double t1056 = 560 * psi * t24 * t266 * t73;
        double t1057 = 1200 * t24 * t266 * t36 * t73;
        double t1058 = -1760 * t153 * t24 * t266 * t73;
        double t1059 = -464 * t24 * t298;
        double t1060 = 312 * psi * t24 * t298;
        double t1061 = 760 * t24 * t298 * t36;
        double t1062 = -624 * t153 * t24 * t298;
        double t1063 = 16 * t159 * t24 * t298;
        double t1064 = 400 * t22 * t24 * t298;
        double t1065 = 320 * psi * t22 * t24 * t298;
        double t1066 = -1840 * t22 * t24 * t298 * t36;
        double t1067 = 1120 * t153 * t22 * t24 * t298;
        double t1068 = -136 * psi * t100;
        double t1069 = 126 * t100 * t36;
        double t1070 = 50 * t100 * t153;
        double t1071 = 204 * t100 * t3;
        double t1072 = 480 * psi * t100 * t3;
        double t1073 = -800 * t100 * t3 * t36;
        double t1074 = 100 * t100 * t153 * t3;
        double t1075 = -348 * t100 * t161;
        double t1076 = -120 * psi * t100 * t161;
        double t1077 = 678 * t100 * t161 * t36;
        double t1078 = -198 * t100 * t153 * t161;
        double t1079 = 180 * t100 * t167;
        double t1080 = -480 * psi * t100 * t167;
        double t1081 = 420 * t100 * t167 * t36;
        double t1082 = -120 * t100 * t153 * t167;
        double t1083 = 192 * psi * t100 * t22;
        double t1084 = -48 * t100 * t22 * t36;
        double t1085 = -144 * t100 * t153 * t22;
        double t1086 = -896 * psi * t100 * t22 * t3;
        double t1087 = 752 * t100 * t22 * t3 * t36;
        double t1088 = 144 * t100 * t153 * t22 * t3;
        double t1089 = 960 * psi * t100 * t161 * t22;
        double t1090 = -1120 * t100 * t161 * t22 * t36;
        double t1091 = 160 * t100 * t153 * t161 * t22;
        double t1092 = 264 * t100 * t5;
        double t1093 = 664 * psi * t100 * t5;
        double t1094 = -352 * t100 * t36 * t5;
        double t1095 = -600 * t100 * t153 * t5;
        double t1096 = 24 * t100 * t159 * t5;
        double t1097 = -1008 * t100 * t3 * t5;
        double t1098 = -576 * psi * t100 * t3 * t5;
        double t1099 = 1528 * t100 * t3 * t36 * t5;
        double t1100 = 112 * t100 * t153 * t3 * t5;
        double t1101 = -56 * t100 * t159 * t3 * t5;
        double t1102 = 840 * t100 * t161 * t5;
        double t1103 = -2040 * psi * t100 * t161 * t5;
        double t1104 = 1560 * t100 * t161 * t36 * t5;
        double t1105 = -360 * t100 * t153 * t161 * t5;
        double t1106 = -1024 * psi * t100 * t22 * t5;
        double t1107 = -272 * t100 * t22 * t36 * t5;
        double t1108 = 1296 * t100 * t153 * t22 * t5;
        double t1109 = 2560 * psi * t100 * t22 * t3 * t5;
        double t1110 = -1520 * t100 * t22 * t3 * t36 * t5;
        double t1111 = -1040 * t100 * t153 * t22 * t3 * t5;
        double t1112 = -624 * t100 * t266;
        double t1113 = -544 * psi * t100 * t266;
        double t1114 = -136 * t100 * t266 * t36;
        double t1115 = 1336 * t100 * t153 * t266;
        double t1116 = -32 * t100 * t159 * t266;
        double t1117 = 1200 * t100 * t266 * t3;
        double t1118 = -2400 * psi * t100 * t266 * t3;
        double t1119 = 1200 * t100 * t266 * t3 * t36;
        double t1120 = 1280 * psi * t100 * t22 * t266;
        double t1121 = 1440 * t100 * t22 * t266 * t36;
        double t1122 = -2720 * t100 * t153 * t22 * t266;
        double t1123 = 480 * t100 * t298;
        double t1124 = -480 * psi * t100 * t298;
        double t1125 = -480 * t100 * t298 * t36;
        double t1126 = 480 * t100 * t153 * t298;
        double t1127 = 144 * psi * t491;
        double t1128 = -144 * t36 * t491;
        double t1129 = -672 * psi * t3 * t491;
        double t1130 = 960 * t3 * t36 * t491;
        double t1131 = -288 * t153 * t3 * t491;
        double t1132 = 720 * psi * t161 * t491;
        double t1133 = -1200 * t161 * t36 * t491;
        double t1134 = 480 * t153 * t161 * t491;
        double t1135 = -768 * psi * t491 * t5;
        double t1136 = 480 * t36 * t491 * t5;
        double t1137 = 288 * t153 * t491 * t5;
        double t1138 = 1920 * psi * t3 * t491 * t5;
        double t1139 = -2400 * t3 * t36 * t491 * t5;
        double t1140 = 480 * t153 * t3 * t491 * t5;
        double t1141 = 960 * psi * t266 * t491;
        double t1142 = -960 * t153 * t266 * t491;
        double t1143 =
            -8 + t1000 + t1001 + t1002 + t1003 + t1004 + t1005 + t1006 + t1007 + t1008 +
            t1009 + t101 + t1010 + t1011 + t1012 + t1013 + t1014 + t1015 + t1016 + t1017 +
            t1018 + t1019 + t1020 + t1021 + t1022 + t1023 + t1024 + t1025 + t1026 +
            t1027 + t1028 + t1029 + t1030 + t1031 + t1032 + t1033 + t1034 + t1035 +
            t1036 + t1037 + t1038 + t1039 + t1040 + t1041 + t1042 + t1043 + t1044 +
            t1045 + t1046 + t1047 + t1048 + t1049 + t1050 + t1051 + t1052 + t1053 +
            t1054 + t1055 + t1056 + t1057 + t1058 + t1059 + t1060 + t1061 + t1062 +
            t1063 + t1064 + t1065 + t1066 + t1067 + t1068 + t1069 + t1070 + t1071 +
            t1072 + t1073 + t1074 + t1075 + t1076 + t1077 + t1078 + t1079 + t1080 +
            t1081 + t1082 + t1083 + t1084 + t1085 + t1086 + t1087 + t1088 + t1089 +
            t1090 + t1091 + t1092 + t1093 + t1094 + t1095 + t1096 + t1097 + t1098 +
            t1099 + t1100 + t1101 + t1102 + t1103 + t1104 + t1105 + t1106 + t1107 +
            t1108 + t1109 + t1110 + t1111 + t1112 + t1113 + t1114 + t1115 + t1116 +
            t1117 + t1118 + t1119 + t1120 + t1121 + t1122 + t1123 + t1124 + t1125 +
            t1126 + t1127 + t1128 + t1129 + t1130 + t1131 + t1132 + t1133 + t1134 +
            t1135 + t1136 + t1137 + t1138 + t1139 + t1140 + t1141 + t1142 + t154 + t160 +
            t166 + t172 + t176 + t181 + t186 + t191 + t195 + t199 + t203 + t207 + t211 +
            t214 + t217 + t317 + t322 + t327 + t332 + t337 + t342 + t347 + t431 + t436 +
            t441 + t522 + t551 + t67 + t677 + t718 + t74 + t833 + t834 + t835 + t836 +
            t837 + t838 + t839 + t840 + t841 + t842 + t843 + t844 + t845 + t846 + t847 +
            t848 + t849 + t850 + t851 + t852 + t853 + t854 + t855 + t856 + t857 + t858 +
            t859 + t86 + t860 + t861 + t862 + t863 + t864 + t865 + t866 + t867 + t868 +
            t869 + t870 + t871 + t872 + t873 + t874 + t875 + t876 + t877 + t878 + t879 +
            t880 + t881 + t882 + t883 + t884 + t885 + t886 + t887 + t888 + t889 + t890 +
            t891 + t892 + t893 + t894 + t895 + t896 + t897 + t898 + t899 + t900 + t901 +
            t902 + t903 + t904 + t905 + t906 + t907 + t908 + t909 + t91 + t910 + t911 +
            t912 + t913 + t914 + t915 + t916 + t917 + t918 + t919 + t920 + t921 + t922 +
            t923 + t924 + t925 + t926 + t927 + t928 + t929 + t930 + t931 + t932 + t933 +
            t934 + t935 + t936 + t937 + t938 + t939 + t940 + t941 + t942 + t943 + t944 +
            t945 + t946 + t947 + t948 + t949 + t950 + t951 + t952 + t953 + t954 + t955 +
            t956 + t957 + t958 + t959 + t960 + t961 + t962 + t963 + t964 + t965 + t966 +
            t967 + t968 + t969 + t970 + t971 + t972 + t973 + t974 + t975 + t976 + t977 +
            t978 + t979 + t980 + t981 + t982 + t983 + t984 + t985 + t986 + t987 + t988 +
            t989 + t990 + t991 + t992 + t993 + t994 + t995 + t996 + t997 + t998 + t999;
        double t1144 = (3 * t1143 * t511) / 2.;
        // c[0] = -1 + t3 + t9;
        // c[1] = -(psi * (-1 + 2 * t3 + 3 * t5));
        // c[2] = 1 + t11 + t12 + t13 + t14 + 3 * psi * t5;
        // c[3] = t18 + psi * t5;
        // c[4] = t21;
        c[0] = t29 * (-1 + t11 + 2 * psi * t22 + 4 * psi * t24 + t3 - psi * t3 -
                      2 * psi * t5 + t9);
        c[1] = -(psi * (-1 + 2 * t22 + 3 * t24));
        c[2] = t29 * (1 + t12 + t14 - 2 * psi * t24 + psi * t3 + 2 * t22 * t36 +
                      t3 * t36 + t37 + t44 + 2 * psi * t5 + t36 * t5);
        c[3] = t18 + psi * t24;
        c[4] = t21;
        c[5] = (psi * t50 * t52 * t62) / 2.;
        c[6] =
            (t62 *
             (-8 + 2 * psi + 36 * psi * t100 + t101 - 9 * psi * t22 - 21 * psi * t24 +
              30 * psi * t22 * t24 + 12 * t3 + 60 * t100 * t3 - 60 * psi * t100 * t3 -
              22 * t22 * t3 + 11 * psi * t22 * t3 - 54 * t24 * t3 + 27 * psi * t24 * t3 +
              50 * t22 * t24 * t3 - 50 * psi * t22 * t24 * t3 + t22 * t36 + 3 * t3 * t36 -
              3 * t22 * t3 * t36 - 6 * t24 * t3 * t36 + t37 + t44 + 28 * t5 -
              10 * psi * t5 + 120 * t100 * t5 - 120 * psi * t100 * t5 - 48 * t22 * t5 +
              34 * psi * t22 * t5 - 116 * t24 * t5 + 78 * psi * t24 * t5 +
              100 * t22 * t24 * t5 - 100 * psi * t22 * t24 * t5 + 2 * t36 * t5 -
              2 * t22 * t36 * t5 - 4 * t24 * t36 * t5 + t67 + 6 * psi * t73 +
              10 * t3 * t73 - 10 * psi * t3 * t73 + 20 * t5 * t73 - 20 * psi * t5 * t73 +
              t74 + t86 + t91)) /
            2.;
        c[7] =
            1 + 3 * t17 * t24 + (t36 * t50 * t52 * t62) / 2. +
            t62 * (-3 * psi + t13 - 8 * t22 + 10 * psi * t22 + 7 * psi * t24 +
                   18 * t22 * t24 - 18 * psi * t22 * t24 + 12 * t22 * t3 -
                   14 * psi * t22 * t3 - 9 * psi * t24 * t3 - 30 * t22 * t24 * t3 +
                   30 * psi * t22 * t24 * t3 + 12 * psi * t5 + 28 * t22 * t5 -
                   36 * psi * t22 * t5 - 26 * psi * t24 * t5 - 60 * t22 * t24 * t5 +
                   60 * psi * t22 * t24 * t5 + 6 * t73 - 6 * psi * t73 - 10 * t3 * t73 +
                   10 * psi * t3 * t73 - 20 * t5 * t73 + 20 * psi * t5 * t73);
        c[8] = -(t17 * t22) + (psi *
                               (3 - 2 * t22 - 7 * t24 - 3 * t3 + 2 * t22 * t3 +
                                9 * t24 * t3 - 12 * t5 + 8 * t22 * t5 + 26 * t24 * t5) *
                               t62) /
                                  2.;
        c[9] = (1 + t11) * t24;
        c[10] = 0;
        c[11] = -1 + t1144 + t511 * t831;
        c[12] = 1 - 3 * t1143 * t511 - (3 * t511 * t831) / 2.;
        c[13] = t1144;
        c[14] = (t511 * t831) / 2.;
        for (auto&& v : c) v /= h;

        if (right) {
            for (auto&& v : c) v *= -1;
            ranges::reverse(c);
        }

        return c;
    }

    void nbs_neumann(real, real, std::span<real>, std::span<real>, bool) const {}
};

stencil make_E2_1(std::span<const real> alpha) { return E2_1{alpha}; }

} // namespace ccs::stencils
