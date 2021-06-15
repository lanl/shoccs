#include "laplacian.hpp"

namespace ccs
{

laplacian::laplacian(const mesh& m,
                     const stencil& st,
                     const bcs::Grid& grid_bcs,
                     const bcs::Object& obj_bcs)
    : dx{0, m, st, grid_bcs, obj_bcs},
      dy{1, m, st, grid_bcs, obj_bcs},
      dz{2, m, st, grid_bcs, obj_bcs},
      ex{m.extents()}
{
}

// when there are no neumann conditions in the problem
std::function<void(scalar_span)> laplacian::operator()(scalar_view u) const
{
    return [this, u](scalar_span du) {
        du = 0;
        // accumulate results into du * WRONG * The block matrix does not accumulate
        if (ex[0] > 1) dx(u, du, plus_eq);
        if (ex[1] > 1) dy(u, du, plus_eq);
        if (ex[2] > 1) dz(u, du, plus_eq);
    };
}

std::function<void(scalar_span)> laplacian::operator()(scalar_view u,
                                                       scalar_view nu) const
{
    return [this, u, nu](scalar_span du) {
        du = 0;
        // accumulate results into du
        if (ex[0] > 1) dx(u, nu, du, plus_eq);
        if (ex[1] > 1) dy(u, nu, du, plus_eq);
        if (ex[2] > 1) dz(u, nu, du, plus_eq);
    };
}
} // namespace ccs
