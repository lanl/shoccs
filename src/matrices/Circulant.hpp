#pragma once

#include "common.hpp"

namespace ccs::matrix
{

class Circulant : public Common
{
    std::span<const real> v;

public:
    Circulant() = default;

    Circulant(integer rows, std::span<const real> coeffs);

    Circulant(integer rows,
              integer row_offset,
              integer stride,
              std::span<const real> coeffs);

    integer size() const noexcept { return v.size(); }

    void operator()(std::span<const real> x, std::span<real> b) const;
};

} // namespace ccs::matrix