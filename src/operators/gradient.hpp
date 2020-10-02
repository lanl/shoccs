#pragma once

#include "derivative.hpp"
#include "fields/fields.hpp"

#include <iostream>

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

    decltype(auto) solid_points() const { return dxyz.solid_points(); }

    template <int N, Vector_Field Boundary_Values, Vector_Field Deriv_Values>
    void operator()(const scalar_field<real, N>& f,
                    const scalar_field<real, N>& df,
                    Boundary_Values&& f_bvalues,
                    Deriv_Values&& df_bvalues,
                    vector_field<real>& grad)
    {
        // copy and transpose scalar to all directions
        std::cout << "in grad\nf\t" << (f | vs::take(10)) << '\n';
        f_work = f;
        df_work = df;

        std::cout << "f_work.x\t" << (f_work.x | vs::take(10)) << '\n';
        std::cout << "df_work.x\t" << (df_work.x | vs::take(10)) << '\n';

        // set boundary values
        f_work >> select(dxyz.solid_points()) <<= f_bvalues;
        df_work >> select(dxyz.solid_points()) <<= df_bvalues;

        std::cout << "f_work.x\t" << (f_work.x | vs::take(10)) << '\n';
        std::cout << "df_work.x\t" << (df_work.x | vs::take(10)) << '\n';

        // compute directional derivatives in all directions and store result in `grad`
        dxyz(f_work, df_work, grad);

        std::cout << "grad.x\t" << (grad.x | vs::take(10)) << '\n';
        std::cout << "grad.y\t" << (grad.y | vs::take(10)) << '\n';
        std::cout << "grad.z\t" << (grad.z | vs::take(10)) << '\n';

        auto sz = f_bvalues.size();

        f_bvalues <<= grad >> select(dxyz.solid_points() >> vector_take(sz));

        std::cout << "grad.x\t" << (grad.x | vs::take(10)) << '\n';
        std::cout << "grad.y\t" << (grad.y | vs::take(10)) << '\n';
        std::cout << "grad.z\t" << (grad.z | vs::take(10)) << '\n';
    }
};
} // namespace ccs::op