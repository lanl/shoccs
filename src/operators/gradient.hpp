#pragma once

#include "derivative.hpp"
#include "fields/vector.hpp"

#include <spdlog/spdlog.h>

namespace ccs
{

class gradient
{
    std::shared_ptr<spdlog::logger> logger;
    derivative dx;
    derivative dy;
    derivative dz;
    index_extents ex;

public:
    gradient() = default;

    gradient(const mesh&, const stencil&, const bcs::Grid&, const bcs::Object&);
    gradient(const mesh&,
             const stencil&,
             const bcs::Grid&,
             const bcs::Object&,
             const std::string&);

    std::function<void(vector_span)> operator()(scalar_view) const;
};
} // namespace ccs
