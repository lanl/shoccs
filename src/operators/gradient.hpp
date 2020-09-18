#pragma once

#include "directional.hpp"

namespace ccs::operator
{
    // computes the gradient of a scalar field, returning a vector field
    // should this return a vector_range?
    class gradient
    {
        directional x;
        directional y;
        directional z;
        vector_field<real> v_work; // value workspace
        vector_field<real> d_work; // derivative workspace

    public:
        template <int N,
                  ranges::input_range BV,
                  ranges::random_access_range DV = decltype(ranges::views::empty<real>),
                  ranges::input_range NBV = decltype(ranges::views::empty<real>)>
        void operator()(const scalar_field<real, N>& f,
                        const scalar_field<real, N>& df,
                        vector_field<real>& grad)
        {
            // transpose(f, v_work);
            // transpose(df, d_work);
            // or we can make use of assignment between scalar and vector fields such that the transpose
            // is implicit. I like this.
            v_work = f;
            d_work = df;

            // v_pts and d_pts are something we can own.  v_values and d_values should be
            // passed in
            // v_work.set(v_pts, v_values);
            // d_work.set(d_pts, d_values);

            // what about?  what types should `points` and `values` be?  
            // should they be tuples of ranges?  Custom struct like what was done
            // for trying to capture by value or reference?
            v_work(v_pts) = v_values;
            d_work(d_pts) = d_values;

            // do the work
            x(v_work.x, d_work.x, grad.x);
            y(v_work.y, d_work.y, grad.y);
            z(v_work.z, d_work.z, grad.z);

            // or maybe
            dxyz(v_work, d_work, grad);
        }
    }
} // namespace ccs::operator