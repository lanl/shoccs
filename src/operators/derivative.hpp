#pragma once

#include "directional.hpp"
#include "fields/fields.hpp"

namespace ccs::op
{

class derivative
{
    directional x;
    directional y;
    directional z;

public:
    derivative() = default;

    derivative(directional&& x, directional&& y, directional&& z)
        : x{std::move(x)}, y{std::move(y)}, z{std::move(z)}
    {
    }

    vector_field_index solid_points() const
    {
        return {x.solid_points(), y.solid_points(), z.solid_points()};
    }

    void operator()(const vector_field<real>& f,
                    const vector_field<real>& df,
                    vector_field<real>& dxyz) const;
};
} // namespace ccs::op