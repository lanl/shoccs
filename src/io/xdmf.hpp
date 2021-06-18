#include "index_extents.hpp"
#include "mesh/mesh_types.hpp"

#include <iostream>

namespace ccs::xdmf
{
std::iostream& write(std::iostream&,
                     index_extents,
                     const domain_extents&,
                     int grid_number,
                     real time,
                     std::span<const std::string> var_names,
                     std::span<const std::string> file_names);
}
