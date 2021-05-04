#include "laplacian.hpp"

namespace ccs
{

laplacian::laplacian(const mesh& m,
                     const stencil& st,
                     const bcs::Grid& grid_bcs,
                     const bcs::Object& obj_bcs)
    : dx{0, m, st, grid_bcs, obj_bcs},
      dy{1, m, st, grid_bcs, obj_bcs},
      dz{2, m, st, grid_bcs, obj_bcs}
{
}

// when there are no neumann conditions in the problem
std::function<void(scalar_span)> laplacian::operator()(scalar_view u) const
{
    return [this, u](scalar_span du) {
        du = 0;
        // accumulate results into du * WRONG * The block matrix does not accumulate
        dx(u, du, plus_eq);
        dy(u, du, plus_eq);
        dz(u, du, plus_eq);
    };
}

std::function<void(scalar_span)> laplacian::operator()(scalar_view u,
                                                       scalar_view nu) const
{
    return [this, u, nu](scalar_span du) {
        du = 0;
        // accumulate results into du
        dx(u, nu, du, plus_eq);
        dy(u, nu, du, plus_eq);
        dz(u, nu, du, plus_eq);
    };
}
} // namespace ccs