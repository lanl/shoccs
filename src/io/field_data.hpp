#pragma once

#include "fields/scalar.hpp"
#include "index_extents.hpp"
#include "mesh/mesh_types.hpp"

namespace ccs
{

class field_data
{
    index_extents ix;

public:
    field_data() = default;
    field_data(index_extents ix) : ix{MOVE(ix)} {}

    void write(std::span<const scalar_view> scalars,
               std::span<const std::string> filenames) const;

    void write_geom(std::span<const std::string> filenames,
                    std::array<std::span<const mesh_object_info>, 3>) const;
};
} // namespace ccs
