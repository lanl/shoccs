#include "divergence.hpp"

namespace ccs::op
{

divergence::divergence(derivative&& dxyz, int3 extents)
    : dxyz{std::move(dxyz)}, f_work{extents}, df_work{extents}, tmp{extents}
{
}

divergence::divergence(const stencil& st,
                       const mesh& m,
                       const geometry& g,
                       const grid_boundaries& grid_b,
                       const object_boundaries& obj_b)
    : dxyz{directional{0, st, m, g, grid_b[0], obj_b},
           directional{1, st, m, g, grid_b[1], obj_b},
           directional{2, st, m, g, grid_b[2], obj_b}},
      f_work{m.extents()},
      df_work{m.extents()},
      tmp{m.extents()}
{
}
} // namespace ccs::op