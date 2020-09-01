#pragma once

#include "types.hpp"
#include "indexing.hpp"

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
class mesh
{
    real3 min_;
    real3 max_;
    real3 h_;
    int3 n_;
    int dims_;

public:
    mesh() = default;

    mesh(span<const real> min, span<const real> max, span<const int> n);

    constexpr int dims() const { return dims_; }

    constexpr int size() const { return n_[0]*n_[1]*n_[2]; }    

    constexpr int plane_size(int i) const {
        auto [f, s] = index::dirs(i);
        return n_[f]*n_[s];
    }

    constexpr umesh_line line(int i) const
    {
        assert(i >= 0 && i <= 2);
        return {min_[i], max_[i], h_[i], n_[i]};
    }

    constexpr int3 n_ijk() const {
        return n_;
    }

    // return n in order slow/fast/dir
    constexpr int3 n_dir(int dir) const {
        auto [f, s] = index::dirs(dir);
        return {n_[s], n_[f], n_[dir]};
    }

    // return a function that will take a point in ijk and return the unique
    // coordinate according to dir
    constexpr auto ucf_ijk2dir(int dir) const {
        auto [f, s] = index::dirs(dir);
        return [s, f, dir, nf=n_[f], n=n_[dir]](auto&& ijk) {
            return n * (nf * ijk[s] + ijk[f]) + ijk[dir];
        };
    } 

    // given a point in ijk, compute the unique coordinate according to dir
    constexpr int uc_ijk2dir(int dir, const int3& ijk) {
        return ucf_ijk2dir(dir)(ijk);
    }

    // return a function that will take a point in "dir" space and return the
    // unique coordinate in that space
    constexpr auto ucf_dir(int dir) const {
        return [n=n_dir(dir)](auto&& pt) {
            auto [ns, nf, nd] = n;
            auto [s, f, d] = pt;
            return nd * (nf * s + f) + d;
        };
    }

    constexpr int uc_dir(int dir, const int3& pt) {
        return ucf_dir(dir)(pt);
    }
};
} // namespace ccs