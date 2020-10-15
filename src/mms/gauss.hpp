#pragma once

#include "types.hpp"
#include <span>
#include <vector>

namespace ccs::detail
{
// all gauss mms have the same data and constructors so move them to a seperate class
struct gauss {
    std::vector<real3> center;
    std::vector<real3> variance;
    std::vector<real> amplitude;
    std::vector<real> frequency;

    gauss() = default;

    gauss(std::span<const real3> center,
          std::span<const real3> variance,
          std::span<const real> amplitude,
          std::span<const real> frequency)
        : center{center.begin(), center.end()},
          variance{variance.begin(), variance.end()},
          amplitude{amplitude.begin(), amplitude.end()},
          frequency{frequency.begin(), frequency.end()}
    {
    }
};
} // namespace ccs::detail