#pragma once

#include <optional>
#include <sol/forward.hpp>
#include <variant>

#include "empty_integrator.hpp"
#include "euler.hpp"
#include "io/logging.hpp"
#include "rk4.hpp"
#include "types.hpp"

namespace ccs
{
// forward decl
class system;

class integrator
{
    std::variant<integrators::empty, integrators::rk4, integrators::euler> v;
    using v_t = decltype(v);

public:
    integrator() = default;

    template <typename T>
        requires(std::constructible_from<v_t, T>)
    integrator(T&& t) : v{FWD(t)} {}

    void operator()(system& sys, sim_registry& reg,
                    field_ref u0, field_ref output,
                    field_ref scratch1, field_ref scratch2,
                    const step_controller& ctrl, real dt);

    static std::optional<integrator> from_lua(const sol::table&, const logs& = {});
};

} // namespace ccs
