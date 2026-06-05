#pragma once

#include <array>
#include <span>

#include "index_extents.hpp"
#include "logging.hpp"
#include "mesh/mesh_types.hpp"

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
               std::span<const std::string> file_names,
               std::array<std::span<const mesh_object_info>, 3>,
               const logs& logger) const;
};

} // namespace ccs
