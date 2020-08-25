#pragma once

#include "types.hpp"
#include <vector>

#include <range/v3/range/concepts.hpp>
#include <range/v3/view/sliding.hpp>
#include <range/v3/view/transform.hpp>

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

private:
    template <ranges::random_access_range R>
    friend constexpr auto operator*(const csr& mat, R&& rng)
    {
        //assert(rng.size() >= static_cast<unsigned>(mat.columns()));

        return mat.u | ranges::views::sliding(2) |
               ranges::views::transform([&mat, &rng](auto&& cols) {
                   real acc{};

                   for (int i = cols[0]; i < cols[1]; i++)
                       acc += mat.w[i] * rng[mat.v[i]];

                   return acc;
               });
    }
};
} // namespace ccs::matrix