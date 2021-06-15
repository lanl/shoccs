#pragma once

#include <functional>
#include <optional>
#include <sol/forward.hpp>
#include <variant>

#include "empty_integrator.hpp"
#include "euler.hpp"
#include "fields/field.hpp"
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

    std::function<void(field_span)>
    operator()(system&, const field&, const step_controller&, real dt);

    static std::optional<integrator> from_lua(const sol::table&);
};

} // namespace ccs
