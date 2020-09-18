#pragma once

#include "directional.hpp"

namespace ccs::operator
{
    // compute the divergence of a vector field/
    // should this return a scalar_range?
    class divergence
    {
        directional x;
        directional y;
        directional z;
        vector_field<real> v_work;
        vector_field<real> d_work;
        vector_field<real> tmp;

    public:
        template <int N,
                  ranges::input_range BV,
                  ranges::random_access_range DV = decltype(ranges::views::empty<real>),
                  ranges::input_range NBV = decltype(ranges::views::empty<real>)>
        void operator()(const vector_field<real>& f,
                                           const vector_field<real>& df,
                                           scalar_field<real, N>& div)
        {
            // copy scalar field to work_space and reorder
            v_work = f;
            d_work = df;

            v_work.set(v_pts, v_values);
            d_work.set(d_pts, d_values);

            dxyz(v_work, d_work, tmp);

            div = tmp.x + tmp.y + tmp.z; // if we make use of extents and transposing in the fields/range
            // or mayber
            div = tmp.x;
            div += tmp.y;
            div += tmp.z;


            // should the directional derivatives retuan a scalar_range?
            // but wouldn't the return type be different in 1D vs 2D vs 3D?
            // seems like we really want to just return zeros for some of these but that
            // range of zeros would need explicitly formed via matrix operations rather than just set.
            div = (x(v_work.x, d_work.x) | transpose<N>) +
                  (y(v_work.y, d_work.y) | transpose<N>) +
                  (z(v_work.z, d_work.z) | transpose<N>); 
        }
    };
} // namespace ccs::operator