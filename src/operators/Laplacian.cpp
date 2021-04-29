#include "Laplacian.hpp"

namespace ccs::operators
{
Laplacian::Laplacian(const mesh::Mesh& mesh,
                     const stencils::Stencil& stencil,
                     const bcs::Grid& grid_bcs,
                     const bcs::Object& obj_bcs)
    : dx{0, mesh, stencil, grid_bcs, obj_bcs},
      dy{1, mesh, stencil, grid_bcs, obj_bcs},
      dz{2, mesh, stencil, grid_bcs, obj_bcs}
{
}

// when there are no neumann conditions in the problem
std::function<void(field::ScalarView_Mutable)>
Laplacian::operator()(field::ScalarView_Const u) const
{
    return [this, u](field::ScalarView_Mutable du) {
        du = 0;
        // accumulate results into du * WRONG * The block matrix does not accumulate
        dx(u, du, plus_eq);
        dy(u, du, plus_eq);
        dz(u, du, plus_eq);
    };
}

std::function<void(field::ScalarView_Mutable)>
Laplacian::operator()(field::ScalarView_Const u, field::ScalarView_Const nu) const
{
    return [this, u, nu](field::ScalarView_Mutable du) {
        du = 0;
        // accumulate results into du
        dx(u, nu, du, plus_eq);
        dy(u, nu, du, plus_eq);
        dz(u, nu, du, plus_eq);
    };
}
} // namespace ccs::operators