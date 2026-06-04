#pragma once

#include "inner_block.hpp"
#include "inner_block_meta.hpp"

#include "kokkos_types.hpp"

#include <Kokkos_Graph.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <cassert>
#include <concepts>

namespace ccs::matrix
{
// Block matrix arising from method-of-lines discretization over whole domain.
// Due to the requirements of a cut-cell mesh, the InnerBlocks may not be adjacent to
// eachother.  To simplify construction, a builder class is exposed which computes all
// the zero locations at the end of the construction process
class block
{
    std::vector<inner_block> blocks;

    // Device-accessible arrays for the future TeamPolicy kernel (17c.5).
    // meta_d: one inner_block_meta per line.
    // coeffs_d: all left/circulant/right coefficients concatenated.
    device_view<inner_block_meta*> meta_d;
    device_view<real*> coeffs_d;

    void build_device_arrays()
    {
        if (blocks.empty()) return;

        const int n = static_cast<int>(blocks.size());

        // First pass: compute total coefficient count and populate host metadata.
        int total_coeffs = 0;
        std::vector<inner_block_meta> host_meta(n);
        for (int i = 0; i < n; ++i) {
            const auto& ib = blocks[i];
            const auto& L = ib.left();
            const auto& C = ib.interior_circ();
            const auto& R = ib.right();

            auto& m = host_meta[i];
            m.row_offset = ib.row_offset();
            m.col_offset = ib.col_offset();
            m.stride = ib.stride();
            m.left_rows = L.rows();
            m.left_cols = L.columns();
            m.left_coeff_offset = total_coeffs;
            total_coeffs += L.size();
            m.interior_rows = C.rows();
            m.interior_coeff_offset = total_coeffs;
            m.stencil_width = C.size();
            total_coeffs += C.size();
            m.right_rows = R.rows();
            m.right_cols = R.columns();
            m.right_coeff_offset = total_coeffs;
            m.right_col_offset = R.col_offset();
            total_coeffs += R.size();
        }

        // Allocate device views.
        meta_d = device_view<inner_block_meta*>("block_meta", n);
        coeffs_d = device_view<real*>("block_coeffs", total_coeffs);

        // Copy metadata to device.
        auto h_meta = Kokkos::View<const inner_block_meta*, Kokkos::HostSpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
            host_meta.data(), n);
        Kokkos::deep_copy(meta_d, h_meta);

        // Second pass: copy coefficients into a contiguous host buffer, then to device.
        std::vector<real> host_coeffs(total_coeffs);
        for (int i = 0; i < n; ++i) {
            const auto& ib = blocks[i];
            const auto& m = host_meta[i];

            auto lc = ib.left().data();
            std::copy(lc.begin(), lc.end(), host_coeffs.begin() + m.left_coeff_offset);

            auto cc = ib.interior_circ().data();
            std::copy(cc.begin(), cc.end(), host_coeffs.begin() + m.interior_coeff_offset);

            auto rc = ib.right().data();
            std::copy(rc.begin(), rc.end(), host_coeffs.begin() + m.right_coeff_offset);
        }

        auto h_coeffs = Kokkos::View<const real*, Kokkos::HostSpace,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
            host_coeffs.data(), total_coeffs);
        Kokkos::deep_copy(coeffs_d, h_coeffs);
    }

public:
    block() = default;

    block(std::vector<inner_block>&& blocks) : blocks{std::move(blocks)}
    {
        build_device_arrays();
    }

    integer rows() const
    {
        if (blocks.empty()) return 0;
        const auto& b = blocks.back();
        // doesn't properly handle case when last point in domain is inside an object
        return b.row_offset() + b.rows() * b.stride();
    }

    const device_view<inner_block_meta*>& metadata_view() const { return meta_d; }
    const device_view<real*>& coefficients_view() const { return coeffs_d; }
    int num_lines() const { return static_cast<int>(blocks.size()); }

    // Named functor for the block matvec kernel, shared by operator() and graph_node.
    template <typename Op>
    struct matvec_functor {
        device_view<inner_block_meta*> meta;
        device_view<real*> coeffs;
        const real* x_ptr;
        real* b_ptr;
        Op op;

        using team_policy = Kokkos::TeamPolicy<execution_space>;
        using member_type = typename team_policy::member_type;

        KOKKOS_INLINE_FUNCTION
        void operator()(const member_type& team) const
        {
            const auto m = meta(team.league_rank());
            const int total_rows = m.left_rows + m.interior_rows + m.right_rows;

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, total_rows),
                [&](int local_row) {
                    int out_idx;
                    real dot = 0;

                    if (local_row < m.left_rows) {
                        // Dense left boundary
                        int r = local_row;
                        out_idx = m.row_offset + r * m.stride;
                        Kokkos::parallel_reduce(
                            Kokkos::ThreadVectorRange(team, m.left_cols),
                            [&](int j, real& s) {
                                s += coeffs(m.left_coeff_offset + r * m.left_cols + j)
                                     * x_ptr[m.col_offset + j * m.stride];
                            }, dot);
                    } else if (local_row < m.left_rows + m.interior_rows) {
                        // Circulant interior
                        int r = local_row - m.left_rows;
                        out_idx = m.row_offset + (m.left_rows + r) * m.stride;
                        const int half_w = m.stencil_width / 2;
                        Kokkos::parallel_reduce(
                            Kokkos::ThreadVectorRange(team, m.stencil_width),
                            [&](int j, real& s) {
                                s += coeffs(m.interior_coeff_offset + j)
                                     * x_ptr[out_idx + (j - half_w) * m.stride];
                            }, dot);
                    } else {
                        // Dense right boundary
                        int r = local_row - m.left_rows - m.interior_rows;
                        out_idx = m.row_offset + (m.left_rows + m.interior_rows + r) * m.stride;
                        Kokkos::parallel_reduce(
                            Kokkos::ThreadVectorRange(team, m.right_cols),
                            [&](int j, real& s) {
                                s += coeffs(m.right_coeff_offset + r * m.right_cols + j)
                                     * x_ptr[m.right_col_offset + j * m.stride];
                            }, dot);
                    }

                    Kokkos::single(Kokkos::PerThread(team), [&]() {
                        op(b_ptr[out_idx], dot);
                    });
                });
        }
    };

    template <typename Op = eq_t>
    void operator()(std::span<const real> x, std::span<real> b, Op op = {}) const
    {
        Kokkos::Profiling::ScopedRegion region("block::operator()");
        const auto n = num_lines();
        if (n == 0) return;

        constexpr int vector_len = 8;
        using team_policy = Kokkos::TeamPolicy<execution_space>;

        Kokkos::parallel_for(
            team_policy(n, Kokkos::AUTO, vector_len),
            matvec_functor<Op>{meta_d, coeffs_d, x.data(), b.data(), op});
    }

    // Chain a TeamPolicy graph node that performs the block matvec with the given op.
    // For empty blocks (0 lines), the node executes zero teams.
    template <typename NodeType, typename Op = eq_t>
    auto graph_node(NodeType parent, const real* x_ptr, real* b_ptr, Op op = {}) const
    {
        const auto n = num_lines();

        constexpr int vector_len = 8;
        using team_policy = Kokkos::TeamPolicy<execution_space>;

        return parent.then_parallel_for(
            "block_matvec",
            team_policy(n, Kokkos::AUTO, vector_len),
            matvec_functor<Op>{meta_d, coeffs_d, x_ptr, b_ptr, op});
    }

    void visit(visitor& v) const
    {
        for (auto&& block : blocks) { block.visit(v); }
    }

    struct builder;
};

struct block::builder {
    std::vector<inner_block> b;

    builder() = default;

    builder(integer n) { b.reserve(n); }

    template <typename... Args>
        requires std::constructible_from<inner_block, Args...>
    void add_inner_block(Args&&... args) { b.emplace_back(std::forward<Args>(args)...); }

    block to_block() &&
    {
#ifndef NDEBUG
        // Verify output row ranges are disjoint.
        for (std::size_t i = 0; i < b.size(); ++i) {
            auto lo_i = b[i].row_offset();
            auto hi_i = lo_i + b[i].rows() * b[i].stride();
            for (std::size_t j = i + 1; j < b.size(); ++j) {
                auto lo_j = b[j].row_offset();
                auto hi_j = lo_j + b[j].rows() * b[j].stride();
                assert((hi_i <= lo_j || hi_j <= lo_i) &&
                       "block inner_blocks have overlapping output row ranges");
            }
        }
#endif
        return block{MOVE(b)};
    }
};
} // namespace ccs::matrix
