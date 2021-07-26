#pragma once

#include "fields/field.hpp"
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

    void write(field_view, std::span<const std::string> filenames) const;

    void write_geom(std::span<const std::string> filenames,
                    tuple<std::span<const mesh_object_info>,
                          std::span<const mesh_object_info>,
                          std::span<const mesh_object_info>>) const;
};
} // namespace ccs
