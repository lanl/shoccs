#include "derivative.hpp"

namespace ccs::op
{
void derivative::operator()(const vector_field<real>& f,
                            const vector_field<real>& df,
                            vector_field<real>& dxyz) const
{
    x(f.x, df.x, dxyz.x);
    y(f.y, df.y, dxyz.y);
    z(f.z, df.z, dxyz.z);
}
} // namespace ccs::op