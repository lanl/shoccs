#include "derivative.hpp"
namespace ccs::op
{
void derivative::operator()(const vector_field<real>& f,
                            const vector_field<real>& df,
                            vector_field<real>& dxyz) const
{
    const auto [nx, ny, nz] = f.extents();

    if (nx > 1)
        x(f.x, df.x, dxyz.x);
    else
        dxyz.x = 0;
    if (ny > 1)
        y(f.y, df.y, dxyz.y);
    else
        dxyz.y = 0;
    if (nz > 1)
        z(f.z, df.z, dxyz.z);
    else
        dxyz.z = 0;
}
} // namespace ccs::op