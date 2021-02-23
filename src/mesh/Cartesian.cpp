#include "Cartesian.hpp"

#include <range/v3/algorithm/copy.hpp>
#include <range/v3/algorithm/copy_n.hpp>
#include <range/v3/algorithm/count_if.hpp>
#include <range/v3/range/concepts.hpp>
#include <range/v3/view/zip_with.hpp>

namespace ccs::mesh
{

Cartesian::Cartesian(span<const real> min, span<const real> max, span<const int> n)
{

    rs::copy_n(min.begin(), std::min(min.size(), 3ul), ranges::begin(min_));
    rs::copy_n(max.begin(), std::min(max.size(), 3ul), ranges::begin(max_));
    rs::copy_n(n.begin(), std::min(n.size(), 3ul), ranges::begin(n_));

    // ensure that unspecified dimensions get set to 1
    for (auto&& i : n_) i = std::max(1, i);

    rs::copy(vs::zip_with(
                 [](real mn, real mx, int n) {
                     return (n - 1) ? (mx - mn) / (n - 1)
                                    : std::numeric_limits<real>::max();
                 },
                 min_,
                 max_,
                 n_),
             rs::begin(h_));

    dims_ = rs::count_if(n_, [](int n) { return !!(n - 1); });
}

} // namespace ccs::mesh