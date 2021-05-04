#include "gradient.hpp"

namespace ccs
{
std::function<void(vector_span)> gradient::operator()(scalar_view u)
{
    return std::function<void(vector_span)>{[this, u](vector_span du) {
        dx(u, get<0>(du));
        dy(u, get<1>(du));
        dz(u, get<2>(du));
    }};
}
} // namespace ccs