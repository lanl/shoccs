#pragma once

#include "derivative.hpp"

namespace ccs
{
class laplacian
{
    std::shared_ptr<spdlog::logger> logger;
    derivative dx;
    derivative dy;
    derivative dz;
    index_extents ex;

public:
    laplacian() = default;

    laplacian(const mesh&,
              const stencil&,
              const bcs::Grid&,
              const bcs::Object&,
              const logs& logger = {});

    // when there are no neumann conditions in the problem
    std::function<void(scalar_span)> operator()(scalar_view) const;

    std::function<void(scalar_span)> operator()(scalar_view field_values,
                                                scalar_view derivative_values) const;
};
} // namespace ccs
