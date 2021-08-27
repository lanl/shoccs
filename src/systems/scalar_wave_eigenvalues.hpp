#pragma once

#include "fields/field.hpp"
#include "io/field_io.hpp"
#include "operators/gradient.hpp"
#include "temporal/step_controller.hpp"
#include "types.hpp"

#include <sol/forward.hpp>
#include <spdlog/spdlog.h>

namespace ccs::systems
{

class scalar_wave_eigenvalues
{
    mesh m;
    bcs::Grid grid_bcs;
    bcs::Object object_bcs;

    gradient grad; // field operator

    real3 center; // center of the circular wave
    real radius;

    vector_real grad_G;

    std::shared_ptr<spdlog::logger> logger;

public:
    scalar_wave_eigenvalues() = default;

    scalar_wave_eigenvalues(
        mesh&&, bcs::Grid&&, bcs::Object&&, stencil, real3 center, real radius);

    void operator()(field& s, const step_controller&);

    system_stats stats(const field& u0, const field& u1, const step_controller&) const;

    bool valid(const system_stats&) const;

    real timestep_size(const field&, const step_controller&) const;

    void rhs(field_view, real, field_span);

    void update_boundary(field_span, real time);

    void log(const system_stats& stats, const step_controller& controller);

    bool write(field_io&, field_view, const step_controller&, real);

    real3 summary(const system_stats&) const;

    system_size size() const;

    static std::optional<scalar_wave_eigenvalues> from_lua(const sol::table&);
};
} // namespace ccs::systems
