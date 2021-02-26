#include "Rect.hpp"

namespace ccs
{

template <int I>
static shape
make_rect(int id, const real3& corner0, const real3& corner1, real fluid_normal)
{
    constexpr int S = index::dir<I>::slow;
    constexpr int F = index::dir<I>::fast;
    fluid_normal = fluid_normal > 0 ? 1 : -1;
    return {rect<I>{real2{corner0[S], corner0[F]},
                    real2{corner1[S], corner1[F]},
                    corner0[I],
                    fluid_normal,
                    id}};
}

shape make_xy_rect(int id, const real3& corner0, const real3& corner1, real fluid_normal)
{
    return make_rect<2>(id, corner0, corner1, fluid_normal);
}

shape make_xz_rect(int id, const real3& corner0, const real3& corner1, real fluid_normal)
{
    return make_rect<1>(id, corner0, corner1, fluid_normal);
}

shape make_yz_rect(int id, const real3& corner0, const real3& corner1, real fluid_normal)
{
    return make_rect<0>(id, corner0, corner1, fluid_normal);
}
} // namespace ccs