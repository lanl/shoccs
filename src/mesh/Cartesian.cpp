#include "Cartesian.hpp"

#include <range/v3/algorithm/copy.hpp>
#include <range/v3/algorithm/count_if.hpp>
#include <range/v3/range/concepts.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/linear_distribute.hpp>
#include <range/v3/view/repeat.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/zip_with.hpp>

namespace ccs::mesh
{

Cartesian::Cartesian(span<const real> min, span<const real> max, span<const int> n)
{
    constexpr auto concat_copy = [](auto&& in, auto val, auto&& out) {
        rs::copy(vs::concat(in, vs::repeat(val)) | vs::take(3), rs::begin(out));
    };
    concat_copy(min, 0.0, min_);
    concat_copy(max, null_v<>, max_);
    concat_copy(n, 1, n_);

    // ensure that unspecified dimensions get set to 1
    for (auto&& i : n_) i = std::max(1, i);

    rs::copy(vs::zip_with([](real mn,
                             real mx,
                             int n) { return (n - 1) ? (mx - mn) / (n - 1) : null_v<>; },
                          min_,
                          max_,
                          n_),
             rs::begin(h_));

    dims_ = rs::count_if(n_, [](auto n) { return !!(n - 1); });

    x_ = vs::linear_distribute(min_[0], max_[0], n_[0]) | rs::to<std::vector<real>>();
    y_ = vs::linear_distribute(min_[1], max_[1], n_[1]) | rs::to<std::vector<real>>();
    z_ = vs::linear_distribute(min_[2], max_[2], n_[2]) | rs::to<std::vector<real>>();
}

} // namespace ccs::mesh