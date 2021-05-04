#pragma once

#include "types.hpp"

#include <compare>
#include <range/v3/range/concepts.hpp>
#include <vector>

namespace ccs::matrix
{
class csr
{
    // standard csr format
    std::vector<real> w;    // values
    std::vector<integer> v; // column indices
    std::vector<integer> u; // starting column index for rows

public:
    csr() = default;

    template <ranges::input_range W, ranges::input_range V, ranges::input_range U>
    csr(W&& w, V&& v, U&& u)
        : w(rs::begin(w), rs::end(w)),
          v(rs::begin(v), rs::end(v)),
          u(rs::begin(u), rs::end(u))
    {
    }

    integer rows() const { return u.size() ? u.size() - 1 : 0; }

    // number of non-zero entries
    integer size() const { return (integer)w.size(); }

    void operator()(std::span<const real> x, std::span<real> b) const;

    struct builder;
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
        // std::cout << "adding\t" << row << '\t' << col << '\t' << v << '\n';
        p.emplace_back(row, col, v);
    }

    csr to_csr(integer nrows);
};

// using CSR_Builder = csr::builder_;

} // namespace ccs::matrix