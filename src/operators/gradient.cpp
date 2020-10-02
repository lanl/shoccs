#include "gradient.hpp"

namespace ccs::op
{

gradient::gradient(derivative&& dxyz, int3 extents)
    : dxyz{std::move(dxyz)}, f_work{extents}, df_work{extents}
{
}

gradient::gradient(const stencil& st,
                   const mesh& m,
                   const geometry& g,
                   const grid_boundaries& grid_b,
                   const object_boundaries& obj_b)
    : dxyz{directional{0, st, m, g, grid_b[0], obj_b},
           directional{1, st, m, g, grid_b[1], obj_b},
           directional{2, st, m, g, grid_b[2], obj_b}},
      f_work{m.extents()},
      df_work{m.extents()}
{
}
} // namespace ccs::op