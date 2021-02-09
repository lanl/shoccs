#pragma once

#include "vector_fwd.hpp"

#include "scalar_field.hpp"
#include "vector_field.hpp"

namespace ccs
{
namespace detail
{
template <typename T, typename U, typename V>
struct vector_proxy {
    using S = T;
    T field;
    U obj;
    V m;
};

template <typename T, typename U, typename V>
vector_proxy(T&&, U&&, V&&) -> vector_proxy<T, U, V>;
} // namespace detail

template <typename T>
struct vector {
    using S = vector_field<T>;
    using X = vector_field<T>::X;
    using Y = vector_field<T>::Y;
    using Z = vector_field<T>::Z;
    using V = vector_range<std::vector<T>>;
    using M = vector_range<std::span<const int3>>;
    using Type = vector<T>;

    S field;
    V obj;
    M m;

    vector() = default;

    vector(int3 ex) : field{ex} {}
    vector(int3 ex, int3 b_size)
        : field{ex}, obj{v_arg(b_size[0]), v_arg(b_size[1]), v_arg(b_size[2])}
    {
    }
    template <Vector_Field VF, Vector_Field BF>
    vector(VF&& field, BF&& obj, M m) : field{FWD(field)}, obj{FWD(obj)}, m{MOVE(m)}
    {
    }

    template <Vector U>
    requires(!std::same_as<std::remove_cvref_t<U>, Type>) vector(U&& u)
        : field{FWD(u).field}, obj{FWD(u).obj}, m{FWD(u).m}
    {
    }

    template <Vector U>
    requires(!std::same_as<std::remove_cvref_t<U>, Type>) vector& operator=(U&& u)
    {
        field = FWD(u).field;
        obj = FWD(u).obj;
        m = FWD(u).m;
        return *this;
    }

    int3 extents() const { return field.extents(); }

#define SHOCCS_GEN_OPERATORS_(op, acc)                                                   \
    template <Numeric N>                                                                 \
    vector& op(N n)                                                                      \
    {                                                                                    \
        field acc n;                                                                     \
        obj acc n;                                                                       \
        return *this;                                                                    \
    }

    SHOCCS_GEN_OPERATORS_(operator=, =)

#define SHOCCS_GEN_OPERATORS(op, acc)                                                    \
    SHOCCS_GEN_OPERATORS_(op, acc)                                                       \
                                                                                         \
    template <Vector R>                                                                  \
    vector& op(R&& r)                                                                    \
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
    template <Vector T, Vector U>                                                        \
    requires requires(T t, U u)                                                          \
    {                                                                                    \
        t.field f u.field;                                                               \
        t.obj f u.obj;                                                                   \
    }                                                                                    \
    constexpr auto op(T&& t, U&& u)                                                      \
    {                                                                                    \
        return detail::vector_proxy{                                                     \
            FWD(t).field f FWD(u).field, FWD(t).obj f FWD(u).obj, FWD(t).m};             \
    }                                                                                    \
    template <Vector T, Numeric U>                                                       \
    requires requires(T t, U u)                                                          \
    {                                                                                    \
        t.field f u;                                                                     \
        t.obj f u;                                                                       \
    }                                                                                    \
    constexpr auto op(T&& t, U u)                                                        \
    {                                                                                    \
        return detail::vector_proxy{FWD(t).field f u, FWD(t).obj f u, FWD(t).m};         \
    }                                                                                    \
    template <Vector T, Numeric U>                                                       \
    requires requires(T t, U u)                                                          \
    {                                                                                    \
        u f t.field;                                                                     \
        u f t.obj;                                                                       \
    }                                                                                    \
    constexpr auto op(U u, T&& t)                                                        \
    {                                                                                    \
        return detail::vector_proxy{u f FWD(t).field, u f FWD(t).obj, FWD(t).m};         \
    }

SHOCCS_GEN_OPERATORS(operator+, +)
SHOCCS_GEN_OPERATORS(operator-, -)
SHOCCS_GEN_OPERATORS(operator*, *)
SHOCCS_GEN_OPERATORS(operator/, /)
#undef SHOCCS_GEN_OPERATORS

using vec = vector<real>;

} // namespace ccs