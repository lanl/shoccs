#include "index_extents.hpp"
#include "mesh/mesh_types.hpp"

#include <iostream>

namespace ccs
{

class xdmf
{
    std::string xmf_filename;
    index_extents ix;
    domain_extents bounds;

public:
    xdmf() = default;
    xdmf(std::string xmf_filename, index_extents ix, domain_extents bounds)
        : xmf_filename{MOVE(xmf_filename)}, ix{MOVE(ix)}, bounds{MOVE(bounds)}
    {
    }

    void write(int grid_number,
               real time,
               std::span<const std::string> var_names,
               std::span<const std::string> file_names);
};

} // namespace ccs
