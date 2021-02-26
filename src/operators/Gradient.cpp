#include "Gradient.hpp"

namespace ccs::operators
{
std::function<void(field::VectorView_Mutable)>
Gradient::operator()(field::ScalarView_Const u)
{
    return std::function<void(field::VectorView_Mutable)>{
        [this, u](field::VectorView_Mutable du) {
            dx(u, du.x());
            dy(u, du.y());
            dz(u, du.z());
        }};
}
} // namespace ccs::operators