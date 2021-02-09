#pragma once

#include "scalar_fwd.hpp"

#include "scalar_field.hpp"
#include "vector_field.hpp"

namespace ccs
{

namespace detail
{
template <typename T, typename U, typename V>
struct scalar_proxy {
    using S = T;
    T field;
    U obj;
    V m;
};
template <typename T, typename U, typename V>
scalar_proxy(T&&, U&&, V &&) -> scalar_proxy<T, U, V>;

} // namespace detail

template <typename T, int I>
struct scalar {
    using S = scalar_field<T, I>;
    using V = vector_range<std::vector<T>>;
    using M =
        vector_range<std::span<const int3>>; // mapping for Boundary points into domain
    using Type = scalar<T, I>;

    S field;
    V obj;
    M m;

    scalar() = default;

    scalar(int3 ex) : field{ex} {}
    scalar(int3 ex, int3 b_size)
        : field{ex}, obj{v_arg(b_size[0]), v_arg(b_size[1]), v_arg(b_size[2])}
    {
    }

    template <Scalar_Field SF, Vector_Field VF>
    scalar(SF&& field, VF&& obj, M m) : field{FWD(field)}, obj{FWD(obj)}, m{MOVE(m)}
    {
    }

    template <Scalar U>
    requires(!std::same_as<std::remove_cvref_t<U>, Type>) scalar(U&& u)
        : field{FWD(u).field}, obj{FWD(u).obj}, m{FWD(u).m}
    {
    }

    template <Scalar U>
    requires(!std::same_as<std::remove_cvref_t<U>, Type>) scalar& operator=(U&& u)
    {
        field = FWD(u).field;
        obj = FWD(u).obj;
        m = FWD(u).m;
        return *this;
    }

    int3 extents() const { return field.extents(); }

#define SHOCCS_GEN_OPERATORS_(op, acc)                                                   \
    template <Numeric N>                                                                 \
    scalar& op(N n)                                                                      \
    {                                                                                    \
        field acc n;                                                                     \
        obj acc n;                                                                       \
        return *this;                                                                    \
    }

    SHOCCS_GEN_OPERATORS_(operator=, =)

#define SHOCCS_GEN_OPERATORS(op, acc)                                                    \
    SHOCCS_GEN_OPERATORS_(op, acc)                                                       \
                                                                                         \
    template <Scalar R>                                                                  \
    scalar& op(R&& r)                                                                    \
    {                                                                                    \
        field acc r.field;                                                               \
        obj acc r.obj;                                                                   \
        return *this;                                                                    \
    }

    SHOCCS_GEN_OPERATORS(operator+=, +=)
    SHOCCS_GEN_OPERATORS(operator-=, -=)
    SHOCCS_GEN_OPERATORS(operator*=, *=)
    SHOCCS_GEN_OPERATORS(operator/=, /=)
#undef SHOCCS_GEN_OPERATORS
#undef SHOCCS_GEN_OPERATORS_
};

#define SHOCCS_GEN_OPERATORS(op, f)                                                      \
    template <Scalar T, Scalar U>                                                        \
    requires requires(T t, U u)                                                          \
    {                                                                                    \
        t.field f u.field;                                                               \
        t.obj f u.obj;                                                                   \
    }                                                                                    \
    constexpr auto op(T&& t, U&& u)                                                      \
    {                                                                                    \
        return detail::scalar_proxy{                                                     \
            FWD(t).field f FWD(u).field, FWD(t).obj f FWD(u).obj, FWD(t).m};             \
    }                                                                                    \
    template <Scalar T, Numeric U>                                                       \
    requires requires(T t, U u)                                                          \
    {                                                                                    \
        t.field f u;                                                                     \
        t.obj f u;                                                                       \
    }                                                                                    \
    constexpr auto op(T&& t, U u)                                                        \
    {                                                                                    \
        return detail::scalar_proxy{FWD(t).field f u, FWD(t).obj f u, FWD(t).m};         \
    }                                                                                    \
    template <Scalar T, Numeric U>                                                       \
    requires requires(T t, U u)                                                          \
    {                                                                                    \
        u f t.field;                                                                     \
        u f t.obj;                                                                       \
    }                                                                                    \
    constexpr auto op(U u, T&& t)                                                        \
    {                                                                                    \
        return detail::scalar_proxy{u f FWD(t).field, u f FWD(t).obj, FWD(t).m};         \
    }

SHOCCS_GEN_OPERATORS(operator+, +)
SHOCCS_GEN_OPERATORS(operator-, -)
SHOCCS_GEN_OPERATORS(operator*, *)
SHOCCS_GEN_OPERATORS(operator/, /)
#undef SHOCCS_GEN_OPERATORS

using x_scalar = scalar<real, 0>;
using y_scalar = scalar<real, 1>;
using z_scalar = scalar<real, 2>;
} // namespace ccs