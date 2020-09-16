#pragma once

#include "types.hpp"
#include <compare>
#include <vector>

#include <range/v3/algorithm/sort.hpp>
#include <range/v3/range/concepts.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/sliding.hpp>
#include <range/v3/view/transform.hpp>

#include <iostream>
namespace ccs::matrix
{
class csr
{
    // standard csr format
    // may add information in the future to accelerate computations with lots (mostly)
    // zeros
    std::vector<real> w; // values
    std::vector<int> v;  // column indices
    std::vector<int> u;  // starting column index for rows

public:
    csr() = default;

    template <ranges::input_range W, ranges::input_range V, ranges::input_range U>
    csr(W&& w, V&& v, U&& u)
        : w(ranges::begin(w), ranges::end(w)),
          v(ranges::begin(v), ranges::end(v)),
          u(ranges::begin(u), ranges::end(u))
    {
    }

    int rows() const { return u.size() ? u.size() - 1 : 0; }

    struct builder_ {
        struct pts {
            int row;
            int col;
            real v;

            auto operator<=>(const pts&) const = default;
            bool operator==(const pts& p) const { return p.row == row && p.col == col; }
        };

        std::vector<pts> p;

        builder_() = default;
        builder_(int n) { p.reserve(n); }

        void add_point(int row, int col, real v)
        {
            std::cout << "adding\t" << row << '\t' << col << '\t' << v << '\n';
            p.emplace_back(row, col, v);
        }

        csr to_csr(int nrows)
        {
            std::vector<int> u(nrows + 1);

            ranges::sort(p);
            auto first = p.begin();
            auto last = p.end();

            for (auto&& [i, r] : u | ranges::view::sliding(2) | ranges::view::enumerate) {
                // initialize to an empty row
                r[1] = r[0];
                while (first != last && first->row == static_cast<int>(i)) {
                    ++r[1];
                    ++first;
                }
            }

            return csr{p | ranges::view::transform([](auto&& p_) { return p_.v; }),
                       p | ranges::view::transform([](auto&& p_) { return p_.col; }),
                       u};
        }
    };

    static builder_ builder(int n = 0) { return n ? builder_{n} : builder_{}; }

private:
    template <ranges::random_access_range R>
    friend constexpr auto operator*(const csr& mat, R&& rng)
    {
        // assert(rng.size() >= static_cast<unsigned>(mat.columns()));

        return mat.u | ranges::views::sliding(2) |
               ranges::views::transform([&mat, &rng](auto&& cols) {
                   real acc{};

                   for (int i = cols[0]; i < cols[1]; i++)
                       acc += mat.w[i] * rng[mat.v[i]];

                   return acc;
               });
    }
};

using csr_builder = csr::builder_;

} // namespace ccs::matrix