#include "gradient.hpp"

namespace ccs
{
gradient::gradient(const mesh& m,
                   const stencil& st,
                   const bcs::Grid& grid_bcs,
                   const bcs::Object& obj_bcs)
    : dx{0, m, st, grid_bcs, obj_bcs},
      dy{1, m, st, grid_bcs, obj_bcs},
      dz{2, m, st, grid_bcs, obj_bcs},
      ex{m.extents()}
{
}

std::function<void(vector_span)> gradient::operator()(scalar_view u) const
{
    return std::function<void(vector_span)>{[this, u](vector_span du) {
        du = 0;
        if (ex[0] > 1) dx(u, get<vi::X>(du));
        if (ex[1] > 1) dy(u, get<vi::Y>(du));
        if (ex[2] > 1) dz(u, get<vi::Z>(du));
    }};
}
} // namespace ccs
