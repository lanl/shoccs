#include "field_io.hpp"

namespace ccs
{

field_data::field_data(const std::array<int, 3>& bounds)
    : sz{bounds[0] * bounds[1] * bounds[2]}
{
}

std::ostream& field_data::write(std::ostream& o, const double* data)
{
    return o.write(reinterpret_cast<const char*>(data), sizeof(*data) * sz);
}

} // namespace ccs
