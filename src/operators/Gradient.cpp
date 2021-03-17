#include "Gradient.hpp"

namespace ccs::operators
{
std::function<void(field::VectorView_Mutable)>
Gradient::operator()(field::ScalarView_Const u)
{
    return std::function<void(field::VectorView_Mutable)>{
        [this, u](field::VectorView_Mutable du) {
            dx(u, get<0>(du));
            dy(u, get<1>(du));
            dz(u, get<2>(du));
        }};
}
} // namespace ccs::operators