#pragma once

#include "index_extents.hpp"
#include "indexing.hpp"
#include "types.hpp"

#include <vector>

#include <cassert>

namespace ccs
{

struct umesh_line {
    real min;
    real max;
    real h;
    int n;
};

// simple representation of a uniform mesh
class cartesian : public index_extents
{
    real3 min_;
    real3 max_;
    real3 h_;
    // int3 n_;
    int dims_;
    std::vector<real> x_;
    std::vector<real> y_;
    std::vector<real> z_;

    constexpr const index_extents& as_extents() const { return *this; }
    constexpr index_extents& as_extents() { return *this; }

public:
    cartesian() = default;

    cartesian(span<const int> n, span<const real> min, span<const real> max);

    constexpr int dims() const { return dims_; }

    constexpr integer size() const
    {
        const int3& n_ = as_extents();
        return (integer)n_[0] * (integer)n_[1] * (integer)n_[2];
    }

    constexpr integer plane_size(int i) const
    {
        auto [f, s] = index::dirs(i);
        const int3& n_ = as_extents();
        return n_[f] * (integer)n_[s];
    }

    constexpr umesh_line line(int i) const
    {
        assert(i >= 0 && i <= 2);
        const int3& n_ = as_extents();
        return {min_[i], max_[i], h_[i], n_[i]};
    }

    constexpr std::span<const real> x() const { return x_; }
    constexpr std::span<const real> y() const { return y_; }
    constexpr std::span<const real> z() const { return z_; }

    constexpr real3 h() const { return h_; }
    constexpr real h(int i) const { return h_[i]; }

    constexpr const auto& extents() const { return as_extents(); }

    constexpr int3 n_ijk() const { return as_extents(); }

    constexpr int n(int i) const { return as_extents()[i]; }

    constexpr bool on_boundary(int dim, bool right_wall, const int3& coordinate) const
    {
        return right_wall ? coordinate[dim] == as_extents()[dim] - 1
                          : coordinate[dim] == 0;
    }

    // return n in order slow/fast/dir
    constexpr int3 n_dir(int dir) const
    {
        const auto& n_ = as_extents();
        auto [f, s] = index::dirs(dir);
        return {n_[s], n_[f], n_[dir]};
    }

    // return a function that will take a point in ijk and return the unique
    // coordinate according to dir
    constexpr auto ucf_ijk2dir(int dir) const
    {
        const auto& n_ = as_extents();
        auto [f, s] = index::dirs(dir);
        return [s, f, dir, nf = n_[f], n = n_[dir]](auto&& ijk) {
            return n * (nf * ijk[s] + ijk[f]) + ijk[dir];
        };
    }

    // given a point in ijk, compute the unique coordinate according to dir
    constexpr int uc_ijk2dir(int dir, const int3& ijk) { return ucf_ijk2dir(dir)(ijk); }

    // return a function that will take a point in "dir" space and return the
    // unique coordinate in that space
    constexpr auto ucf_dir(int dir) const
    {
        return [n = n_dir(dir)](auto&& pt) {
            auto [ns, nf, nd] = n;
            auto [s, f, d] = pt;
            return nd * (nf * s + f) + d;
        };
    }

    constexpr int uc_dir(int dir, const int3& pt) { return ucf_dir(dir)(pt); }
};
} // namespace ccs