#pragma once

#include "scalar_fwd.hpp"

#include "scalar_field.hpp"
#include "vector_field.hpp"

namespace ccs
{

namespace detail
{
template <typename T, typename U>
struct scalar_proxy {
    T field;
    U obj;
};
template <typename T, typename U>
scalar_proxy(T&&, U &&) -> scalar_proxy<T, U>;

} // namespace detail

template <typename T, int I>
struct scalar {
    using S = scalar_field<T, I>;
    using V = vector_range<std::vector<T>>;
    using Type = scalar<T, I>;
    static constexpr int dim = I;

    S field;
    V obj;

    scalar() = default;

    scalar(int3 ex) : field{ex} {}
    scalar(int3 ex, int3 b_size) : field{ex} { obj.resize(b_size); }

    template <Scalar_Field SF, Vector_Field VF>
    scalar(SF&& field, VF&& obj)
        : field{std::forward<SF>(field)}, obj{std::forward<VF>(obj)}
    {
    }

    template <Scalar U>
    requires(!std::same_as<std::remove_cvref_t<U>, Type>) scalar(U&& u)
        : field{std::forward<U>(u).field}, obj{std::forward<U>(u).obj}
    {
    }

    template <Scalar U>
    requires(!std::same_as<std::remove_cvref_t<U>, Type>) scalar& operator=(U&& u)
    {
        field = std::forward<U>(u).field;
        obj = std::forward<U>(u).obj;
        return *this;
    }

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
        return detail::scalar_proxy{std::forward<T>(t).field f std::forward<U>(u).field, \
                                    std::forward<T>(t).obj f std::forward<U>(u).obj};    \
    }                                                                                    \
    template <Scalar T, Numeric U>                                                       \
    requires requires(T t, U u)                                                          \
    {                                                                                    \
        t.field f u;                                                                     \
        t.obj f u;                                                                       \
    }                                                                                    \
    constexpr auto op(T&& t, U u)                                                        \
    {                                                                                    \
        return detail::scalar_proxy{std::forward<T>(t).field f u,                        \
                                    std::forward<T>(t).obj f u};                         \
    }                                                                                    \
    template <Scalar T, Numeric U>                                                       \
    requires requires(T t, U u)                                                          \
    {                                                                                    \
        u f t.field;                                                                     \
        u f t.obj;                                                                       \
    }                                                                                    \
    constexpr auto op(U u, T&& t)                                                        \
    {                                                                                    \
        return detail::scalar_proxy{u f std::forward<T>(t).field,                        \
                                    u f std::forward<T>(t).obj};                         \
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