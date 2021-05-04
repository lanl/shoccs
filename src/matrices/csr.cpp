#include "csr.hpp"

#include <range/v3/algorithm/sort.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/sliding.hpp>
#include <range/v3/view/transform.hpp>

namespace ccs::matrix
{

csr csr::builder::to_csr(integer nrows)
{
    std::vector<int> u(nrows + 1);

    rs::sort(p);
    auto first = p.begin();
    auto last = p.end();

    for (auto&& [i, r] : u | vs::sliding(2) | vs::enumerate) {
        // initialize to an empty row
        r[1] = r[0];
        while (first != last && first->row == static_cast<integer>(i)) {
            ++r[1];
            ++first;
        }
    }

    return csr{p | vs::transform([](auto&& p_) { return p_.v; }),
               p | vs::transform([](auto&& p_) { return p_.col; }),
               u};
}

void csr::operator()(std::span<const real> x, std::span<real> b) const
{
    for (integer row = 0; row < rows(); row++)
        for (integer i = u[row]; i < u[row + 1]; i++) b[row] += w[i] * x[v[i]];
}

} // namespace ccs::matrix