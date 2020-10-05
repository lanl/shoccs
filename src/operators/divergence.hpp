#pragma once

#include "derivative.hpp"

namespace ccs::op
{
    // compute the divergence of a vector field/
    // should this return a scalar_range?
    class divergence
    {
        derivative dxyz;
        vector_field<real> f_work;
        vector_field<real> df_work;
        vector_field<real> tmp;

    public:
        divergence() = default;
        divergence(derivative&& dxyz, int3 extents);
        divergence(const stencil&,
                   const mesh&,
                   const geometry&,
                   const grid_boundaries&,
                   const object_boundaries&);

        template <int N, Vector_Field Boundary_Values, Vector_Field Deriv_Values>
        void operator()(const vector_field<real>& f,
                        const vector_field<real>& df,
                        Boundary_Values&& f_bvalues,
                        Deriv_Values&& df_bvalues,
                        scalar_field<real, N>& div)
        {
            // copy to work_space
            f_work = f;
            df_work = df;

            // set boundary values
            f_work >> select(dxyz.solid_points()) <<= f_bvalues;
            df_work >> select(dxyz.solid_points()) <<= df_bvalues;

            dxyz(f_work, df_work, tmp);

            // copy boundary conditions back. Not sure what the best way to
            // handle this is
            auto sz = f_bvalues.size();
            f_bvalues <<= tmp >> select(dxyz.solid_points() >> vector_take(sz));

            // combine the data at fluid points
            div = tmp.x;
            div += tmp.y;
            div += tmp.z;
        }
    };
} // namespace ccs::operator