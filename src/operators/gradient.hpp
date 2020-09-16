#pragma once

#include "directional.hpp"

namespace ccs::operator
{
    class gradient
    {
        directional x;
        directional y;
        directional z;

        public:

         template <ranges::random_access_range V,
              ranges::input_range BV,
              ranges::random_access_range DV = decltype(ranges::views::empty<real>),
              ranges::input_range NBV = decltype(ranges::views::empty<real>)>
        auto operator()()
        {
            // I think xval and xnm need to be passed in as arguments
            auto xval; // apply bcs on solid points for boundary values in x
            auto xnm; // apply bcs on solid points for neumann values
            auto gradx = x(v, xval, nm, xnm);

            auto yval; // apply bcs on solid points for boundary values in x
            auto ynm; // apply bcs on solid points for neumann values
            auto grady = y(v, yval, nm, ynm);

            auto zval; // apply bcs on solid points for boundary values in y
            auto znm; // apply bcs on solid points for neumann values
            auto gradz = z(v, zval, nm, znm);

            // should there be an option to transpose these to a different direction?
            return {gradx, grady, gradz}


            

        }
    }
}