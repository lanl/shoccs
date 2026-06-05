#pragma once

#include "common.hpp"
#include "matrix_visitor.hpp"

#include "kokkos_types.hpp"

#include <Kokkos_Graph.hpp>

#include <compare>
#include <ranges>
#include <vector>

namespace ccs::matrix
{
class csr
{
    // standard csr format
    std::vector<real> w;    // values
    std::vector<integer> v; // column indices
    std::vector<integer> u; // starting column index for rows
    flag f;

public:
    csr() = default;

    template <std::ranges::input_range W, std::ranges::input_range V, std::ranges::input_range U>
    csr(W&& w, V&& v, U&& u, flag row_col_space = 0)
        : w(std::ranges::begin(w), std::ranges::end(w)),
          v(std::ranges::begin(v), std::ranges::end(v)),
          u(std::ranges::begin(u), std::ranges::end(u)),
          f{row_col_space}
    {
    }

    integer rows() const { return u.size() ? u.size() - 1 : 0; }
    std::span<const integer> column_indices(integer row) const;
    std::span<const real> column_coefficients(integer row) const;

    // number of non-zero entries
    integer size() const { return (integer)w.size(); }

    void operator()(std::span<const real> x, std::span<real> b) const;

    // Chain a RangePolicy graph node that performs the CSR matvec (always +=).
    // For 0-row matrices, the node executes zero iterations.
    template <typename NodeType>
    auto graph_node(NodeType parent, const real* x_ptr, real* b_ptr) const
    {
        const auto nr = rows();
        const auto* wp = w.data();
        const auto* vp = v.data();
        const auto* up = u.data();
        return parent.then_parallel_for(
            "csr_matvec",
            Kokkos::RangePolicy<execution_space>(0, nr),
            KOKKOS_LAMBDA(integer row) {
                for (integer i = up[row]; i < up[row + 1]; i++)
                    b_ptr[row] += wp[i] * x_ptr[vp[i]];
            });
    }

    struct builder;

    flag flags() const { return f; }
    void flags(flag f_) { f = f_; }
    void visit(visitor& v) const { v.visit(*this); }
};

struct csr::builder {

    struct pts {
        integer row;
        integer col;
        real v;

        auto operator<=>(const pts&) const = default;
        bool operator==(const pts& p) const { return p.row == row && p.col == col; }
    };

    std::vector<pts> p;

    builder() = default;
    builder(integer n) { p.reserve(n); }

    void add_point(integer row, integer col, real v)
    {
        p.emplace_back(row, col, v);
    }

    csr to_csr(integer nrows);
};

} // namespace ccs::matrix
