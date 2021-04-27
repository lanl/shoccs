#include "Cartesian.hpp"

#include <range/v3/all.hpp>

namespace ccs::mesh
{

Cartesian::Cartesian(span<const int> n, span<const real> min, span<const real> max)
{
    constexpr auto concat_copy = [](auto&& in, auto val, auto&& out) {
        rs::copy(vs::concat(in, vs::repeat(val)) | vs::take(3), rs::begin(out));
    };
    concat_copy(min, 0.0, min_);
    concat_copy(max, null_v<>, max_);

    int3& n_ = as_extents();
    concat_copy(n | vs::transform([](auto&& v) { return v > 0 ? v : 1; }), 1, n_);

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