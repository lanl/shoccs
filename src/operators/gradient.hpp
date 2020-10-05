#pragma once

#include "derivative.hpp"
#include "fields/fields.hpp"

namespace ccs::op
{
// computes the gradient of a scalar field, returning a vector field
// should this return a vector_range?
class gradient
{
    derivative dxyz;
    vector_field<real> f_work;  // value workspace
    vector_field<real> df_work; // derivative workspace

public:
    gradient() = default;
    gradient(derivative&& dxyz, int3 extents);
    gradient(const stencil&,
             const mesh&,
             const geometry&,
             const grid_boundaries&,
             const object_boundaries&);

    template <int N, Vector_Field Boundary_Values, Vector_Field Deriv_Values>
    void operator()(const scalar_field<real, N>& f,
                    const scalar_field<real, N>& df,
                    Boundary_Values&& f_bvalues,
                    Deriv_Values&& df_bvalues,
                    vector_field<real>& grad)
    {
        // copy and transpose scalar to all directions
        f_work = f;
        df_work = df;

        // set boundary values
        f_work >> select(dxyz.solid_points()) <<= f_bvalues;
        df_work >> select(dxyz.solid_points()) <<= df_bvalues;

        // compute directional derivatives in all directions and store result in `grad`
        dxyz(f_work, df_work, grad);

        auto sz = f_bvalues.size();
        f_bvalues <<= grad >> select(dxyz.solid_points() >> vector_take(sz));
    }
};
} // namespace ccs::op