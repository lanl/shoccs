#pragma once

#include "fields/field.hpp"
#include "operators/gradient.hpp"
#include "temporal/step_controller.hpp"
#include "types.hpp"

#include <sol/forward.hpp>
#include <spdlog/spdlog.h>

namespace ccs::systems
{

// the system of pdes to solve is in this class
class scalar_wave
{
    mesh m;
    bcs::Grid grid_bcs;
    bcs::Object object_bcs;

    gradient grad;

    // required data
    // std::vector<double> grad_c;
    // std::vector<double> grad_u;

    real3 center; // center of the circular wave
    real radius;

    vector_real grad_G;
    vector_real du;

    std::vector<std::string> io_names = {"U"};
    std::shared_ptr<spdlog::logger> logger;

public:
    scalar_wave() = default;

    scalar_wave(mesh&&, bcs::Grid&&, bcs::Object&&, stencil, real3 center, real radius);

    void operator()(field& s, const step_controller&);

    system_stats stats(const field& u0, const field& u1, const step_controller&) const;

    bool valid(const system_stats&) const;

    real timestep_size(const field&, const step_controller&) const;

    void rhs(field_view, real, field_span);

    void update_boundary(field_span, real time);

    real3 summary(const system_stats&) const;

    std::span<const std::string> names() const;

    void log(const system_stats&, const step_controller&);

    system_size size() const;

    static std::optional<scalar_wave> from_lua(const sol::table&);
};

} // namespace ccs::systems
