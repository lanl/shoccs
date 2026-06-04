#include "block.hpp"
#include "inner_block_meta.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "random/random.hpp"
#include <vector>

#include <algorithm>
#include <ranges>

#include "fields/lazy_views.hpp"

#include <Kokkos_Core.hpp>

// Custom main: Kokkos must be initialized before parallel_for calls.
int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard kokkos(argc, argv);
    return Catch::Session().run(argc, argv);
}

using namespace ccs;
using Catch::Matchers::Approx;

constexpr auto g = []() { return pick(); };
constexpr auto x2 = [](auto&& x) { return x + x; };

TEST_CASE("Identity")
{
    using T = std::vector<real>;

    T left_c{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    T int_c{1.0};
    T right_c{1, 0, 0, 1};

    auto bld = matrix::block::builder();
    bld.add_inner_block(16,
                        1,
                        1,
                        1,
                        matrix::dense{4, 4, left_c},
                        matrix::circulant{10, int_c},
                        matrix::dense{2, 2, right_c});
    bld.add_inner_block(16,
                        17,
                        17,
                        1,
                        matrix::dense{4, 4, left_c},
                        matrix::circulant{10, int_c},
                        matrix::dense{2, 2, right_c});
    bld.add_inner_block(16,
                        34,
                        34,
                        1,
                        matrix::dense{4, 4, left_c},
                        matrix::circulant{10, int_c},
                        matrix::dense{2, 2, right_c});

    const auto A = MOVE(bld).to_block();

    REQUIRE(A.rows() == 50);
    T x(A.rows());
    std::generate_n(x.begin(), A.rows(), g);
    T b(x.size());

    A(x, b);

    T xx = x;
    // zero locations
    xx[0] = xx[33] = 0;

    REQUIRE_THAT(b, Approx(xx));

    T xx2(xx.size());
    std::ranges::transform(xx, xx2.begin(), x2);
    A(x, b, plus_eq);
    REQUIRE_THAT(b, Approx(xx2));
}

TEST_CASE("Random Boundary")
{
    using T = std::vector<real>;

    const T left_c{
        2.247323503221594,   -5.0275337061235135, -0.9845370624957113, 9.621361506824222,
        -1.4847274599487292, -5.920356924707782,  -2.5821362891416157, 3.4175272453709056,
        8.530925204174977,   6.082313908057927,   -6.47449633704835,   3.2352002101687702,
        4.758308560669512,   1.598859246390333,   0.9341660471574365,  -2.405757184769108,
        -1.0503192473307266, 2.485464032420346,   9.816354187065464,   5.090921023204807};

    const T int_c{0.7763383518842382, 2.749405412018632, -0.7966643678385346};

    const T right_c{-0.12697902221008128,
                    1.4021179149102307,
                    0.30625420155821903,
                    1.288208276267654,
                    1.3345939131355147,
                    -1.4713774812532359};

    auto bld = matrix::block::builder(3);
    const integer cols = 16;

    bld.add_inner_block(cols,
                        1,
                        1,
                        1,
                        matrix::dense{4, 5, left_c},
                        matrix::circulant{10, int_c},
                        matrix::dense{2, 3, right_c});
    bld.add_inner_block(cols,
                        cols + 5,
                        cols + 5,
                        1,
                        matrix::dense{4, 5, left_c},
                        matrix::circulant{10, int_c},
                        matrix::dense{2, 3, right_c});

    const auto A = MOVE(bld).to_block();

    T rhs_{1.1489955608128035,
           -9.641815125514444,
           0.5975150739415511,
           5.321795555035614,
           -5.467954909319733,
           1.4687325444642383,
           8.370366165425736,
           -2.810816575553474,
           1.815504174076466,
           1.1769187743854737,
           -8.870585410511126,
           -0.14181686526567816,
           7.93970270970086,
           0.14183234352470464,
           -8.054425087325441,
           -3.809075460321715};

    T x{0.0};
    x.insert(x.end(), rhs_.begin(), rhs_.end());
    x.insert(x.end(), 4, 0.0);
    x.insert(x.end(), rhs_.begin(), rhs_.end());
    T b(x.size());

    A(x, b);

    T exact_{109.78978122898833,
             32.2780625869145,
             -32.38838457188155,
             33.25158538253831,
             -12.072197714155793,
             -6.87521436567669,
             26.39304084900066,
             -2.6761855166330424,
             1.8718030426410628,
             11.712151684565809,
             -23.362167910509047,
             -13.601765954771285,
             21.606370954127627,
             12.97052379948305,
             -12.477808805315766,
             -4.962089239867884};
    T exact{0.0};
    exact.insert(exact.end(), exact_.begin(), exact_.end());
    exact.insert(exact.end(), 4, 0.0);
    exact.insert(exact.end(), exact_.begin(), exact_.end());

    REQUIRE_THAT(b, Approx(exact));
}

TEST_CASE("strided")
{
    // Assume we have some field of values on an XxY (15 x 3) grid with the stride in Y ==
    // 1 and we want to compute derivative along the 3 x-lines:
    //
    // 2 5 ... 44
    // 1 4 ... 43
    // 0 3 ... 42

    using T = std::vector<real>;

    auto iota15 = std::views::iota(0, 15);
    const T lc(iota15.begin(), iota15.end());    // 3x5 matrix
    const T ic{-2, -1, 0, 1, 2};                // minimum offset of 2 needed
    auto iota17 = std::views::iota(1, 7);
    const T rc(iota17.begin(), iota17.end());    // 2 x 3

    const integer columns = 15;
    const integer stride = 3;

    // Set up strided operator
    const auto A =
        matrix::block{std::vector{matrix::inner_block{columns,
                                                      0,
                                                      0,
                                                      stride,
                                                      matrix::dense(3, 5, lc),
                                                      matrix::circulant(10, ic),
                                                      matrix::dense(2, 3, rc)},
                                  matrix::inner_block{columns,
                                                      1,
                                                      1,
                                                      stride,
                                                      matrix::dense(3, 5, lc),
                                                      matrix::circulant(10, ic),
                                                      matrix::dense(2, 3, rc)},
                                  matrix::inner_block{columns,
                                                      2,
                                                      2,
                                                      stride,
                                                      matrix::dense(3, 5, lc),
                                                      matrix::circulant(10, ic),
                                                      matrix::dense(2, 3, rc)}}};
    auto iota45 = std::views::iota(0, 45);
    const T x(iota45.begin(), iota45.end());
    T b(x.size());
    A(x, b);

    for (integer offset = 0; offset < 3; offset++) {
        // non-strided operator
        const auto AA = matrix::inner_block(columns,
                                            0,
                                            0,
                                            1,
                                            matrix::dense(3, 5, lc),
                                            matrix::circulant(10, ic),
                                            matrix::dense(2, 3, rc));
        auto strided_xx = ccs::stride(std::views::iota(0, 45) | std::views::drop(offset), stride);
        auto taken_xx = strided_xx | std::views::take(15);
        const T xx(taken_xx.begin(), taken_xx.end());
        T bb(xx.size());
        AA(xx, bb);

        auto strided_bp = ccs::stride(b | std::views::drop(offset), stride);
        auto taken_bp = strided_bp | std::views::take(15);
        const T bp(taken_bp.begin(), taken_bp.end());

        REQUIRE_THAT(bp, Approx(bb));
    }
}

TEST_CASE("device metadata arrays")
{
    using T = std::vector<real>;

    // Non-square boundaries: left 4x5, right 2x3, circulant width 3
    const T left_c{
        2.247, -5.027, -0.984, 9.621,
        -1.484, -5.920, -2.582, 3.417,
        8.530, 6.082, -6.474, 3.235,
        4.758, 1.598, 0.934, -2.405,
        -1.050, 2.485, 9.816, 5.090};
    const T int_c{0.776, 2.749, -0.796};
    const T right_c{-0.126, 1.402, 0.306, 1.288, 1.334, -1.471};

    const integer cols = 16;

    // Two inner_blocks with stride=1 at different offsets.
    auto bld = matrix::block::builder(2);
    bld.add_inner_block(cols, 1, 1, 1,
                        matrix::dense{4, 5, left_c},
                        matrix::circulant{10, int_c},
                        matrix::dense{2, 3, right_c});
    bld.add_inner_block(cols, 20, 20, 1,
                        matrix::dense{4, 5, left_c},
                        matrix::circulant{10, int_c},
                        matrix::dense{2, 3, right_c});
    const auto A = MOVE(bld).to_block();

    REQUIRE(A.num_lines() == 2);

    // Copy metadata back to host and verify.
    const auto& meta_view = A.metadata_view();
    REQUIRE(meta_view.extent(0) == 2);

    std::vector<matrix::inner_block_meta> host_meta(2);
    auto h_meta = Kokkos::View<matrix::inner_block_meta*, Kokkos::HostSpace,
                               Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
        host_meta.data(), 2);
    Kokkos::deep_copy(h_meta, meta_view);

    // Expected coefficient sizes per inner_block: 20 (left) + 3 (circ) + 6 (right) = 29
    const int coeffs_per_block = 20 + 3 + 6;

    SECTION("inner_block 0 metadata")
    {
        const auto& m = host_meta[0];
        REQUIRE(m.row_offset == 1);
        REQUIRE(m.col_offset == 1);
        REQUIRE(m.stride == 1);
        REQUIRE(m.left_rows == 4);
        REQUIRE(m.left_cols == 5);
        REQUIRE(m.left_coeff_offset == 0);
        REQUIRE(m.interior_rows == 10);
        REQUIRE(m.interior_coeff_offset == 20);
        REQUIRE(m.stencil_width == 3);
        REQUIRE(m.right_rows == 2);
        REQUIRE(m.right_cols == 3);
        REQUIRE(m.right_coeff_offset == 23);
        // right_col_offset = col_offset + stride * (columns - right_cols) = 1 + 1*(16-3) = 14
        REQUIRE(m.right_col_offset == 14);
    }

    SECTION("inner_block 1 metadata")
    {
        const auto& m = host_meta[1];
        REQUIRE(m.row_offset == 20);
        REQUIRE(m.col_offset == 20);
        REQUIRE(m.stride == 1);
        REQUIRE(m.left_rows == 4);
        REQUIRE(m.left_cols == 5);
        REQUIRE(m.left_coeff_offset == coeffs_per_block);
        REQUIRE(m.interior_rows == 10);
        REQUIRE(m.interior_coeff_offset == coeffs_per_block + 20);
        REQUIRE(m.stencil_width == 3);
        REQUIRE(m.right_rows == 2);
        REQUIRE(m.right_cols == 3);
        REQUIRE(m.right_coeff_offset == coeffs_per_block + 23);
        // right_col_offset = 20 + 1*(16-3) = 33
        REQUIRE(m.right_col_offset == 33);
    }

    SECTION("coefficient data")
    {
        const auto& coeffs_view = A.coefficients_view();
        REQUIRE(coeffs_view.extent(0) == static_cast<std::size_t>(2 * coeffs_per_block));

        std::vector<real> host_coeffs(2 * coeffs_per_block);
        auto h_coeffs = Kokkos::View<real*, Kokkos::HostSpace,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
            host_coeffs.data(), host_coeffs.size());
        Kokkos::deep_copy(h_coeffs, coeffs_view);

        // Check left coefficients for block 0
        for (int i = 0; i < 20; ++i)
            REQUIRE(host_coeffs[i] == Catch::Approx(left_c[i]));

        // Check circulant coefficients for block 0
        for (int i = 0; i < 3; ++i)
            REQUIRE(host_coeffs[20 + i] == Catch::Approx(int_c[i]));

        // Check right coefficients for block 0
        for (int i = 0; i < 6; ++i)
            REQUIRE(host_coeffs[23 + i] == Catch::Approx(right_c[i]));

        // Check left coefficients for block 1 (same data, offset by coeffs_per_block)
        for (int i = 0; i < 20; ++i)
            REQUIRE(host_coeffs[coeffs_per_block + i] == Catch::Approx(left_c[i]));
    }
}

TEST_CASE("device metadata with stride")
{
    using T = std::vector<real>;

    auto iota15 = std::views::iota(0, 15);
    const T lc(iota15.begin(), iota15.end()); // 3x5 matrix
    const T ic{-2, -1, 0, 1, 2};
    auto iota6 = std::views::iota(1, 7);
    const T rc(iota6.begin(), iota6.end()); // 2x3 matrix

    const integer columns = 15;
    const integer stride = 3;

    const auto A = matrix::block{std::vector{
        matrix::inner_block{columns, 0, 0, stride,
                            matrix::dense(3, 5, lc),
                            matrix::circulant(10, ic),
                            matrix::dense(2, 3, rc)},
        matrix::inner_block{columns, 1, 1, stride,
                            matrix::dense(3, 5, lc),
                            matrix::circulant(10, ic),
                            matrix::dense(2, 3, rc)}}};

    REQUIRE(A.num_lines() == 2);

    std::vector<matrix::inner_block_meta> host_meta(2);
    auto h_meta = Kokkos::View<matrix::inner_block_meta*, Kokkos::HostSpace,
                               Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
        host_meta.data(), 2);
    Kokkos::deep_copy(h_meta, A.metadata_view());

    // Block 0: row_offset=0, col_offset=0, stride=3
    // right_col_offset = 0 + 3*(15-3) = 36
    REQUIRE(host_meta[0].row_offset == 0);
    REQUIRE(host_meta[0].col_offset == 0);
    REQUIRE(host_meta[0].stride == 3);
    REQUIRE(host_meta[0].right_col_offset == 36);

    // Block 1: row_offset=1, col_offset=1, stride=3
    // right_col_offset = 1 + 3*(15-3) = 37
    REQUIRE(host_meta[1].row_offset == 1);
    REQUIRE(host_meta[1].col_offset == 1);
    REQUIRE(host_meta[1].stride == 3);
    REQUIRE(host_meta[1].right_col_offset == 37);
}