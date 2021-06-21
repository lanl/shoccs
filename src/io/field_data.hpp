#pragma once

#include "fields/field.hpp"
#include "index_extents.hpp"

namespace ccs
{

class field_data
{
    index_extents ix;

public:
    field_data() = default;
    field_data(index_extents ix) : ix{MOVE(ix)} {}

    void write(const field&, std::span<const std::string> filenames) const;
};
} // namespace ccs
