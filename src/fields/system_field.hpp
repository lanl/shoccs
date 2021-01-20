#pragma once

#include "r_tuple.hpp"

namespace ccs
{
template <typename T = std::vector<real>>
class system_field
{
    std::vector<scalar<T>> fields;

public:
    system_field() = default;

    system_field(int nfields, int3 extents, solid_points);

    template <std::integral... Is>
    auto operator()(Is&&... i) const
    {
        // use `forward_as_tuple` to get a tuple of references
        return std::forward_as_tuple{fields[i]...};
    }

    template <std::integral... Is>
    auto operator()(Is&&... i)
    {
        return std::forward_as_tuple{fields[i]...};
    }
};

} // namespace ccs