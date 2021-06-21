#include "field_data.hpp"
#include <fstream>

namespace ccs
{

void field_data::write(const field& f, std::span<const std::string> filenames) const
{
    unsigned long sz = ix[0] * ix[1] * ix[2] * sizeof(real);

    for (auto&& [fname, sc] : vs::zip(filenames, f.scalars())) {
        std::ofstream o(fname);
        const real* d = get<si::D>(sc).data();

        o.write(reinterpret_cast<const char*>(d), sz);
    }
}

} // namespace ccs
