#pragma once

#include "SystemField_fwd.hpp"
#include "types.hpp"
#include <functional>

namespace ccs::field::lazy
{
template <typename T>
struct ContainerMath;

class ContainerAccess
{
    template <typename T>
    friend class ContainerMath;
};

#define SHOCCS_GEN_OPERATORS(op, f)                                                      \
    template <typename U, Numeric V>                                                     \
    requires std::derived_from<std::remove_cvref_t<U>, T> friend constexpr auto op(      \
        U&& u, V)                                                                        \
    {                                                                                    \
        return u;                                                                        \
    }                                                                                    \
                                                                                         \
    template <typename U, typename V>                                                    \
    requires std::derived_from<std::remove_cvref_t<U>, T> friend constexpr auto op(      \
        U&& u, V&&)                                                                      \
    {                                                                                    \
        return u;                                                                        \
    }

template <typename T>
struct ContainerMath {

private:
    SHOCCS_GEN_OPERATORS(operator+=, +=)
    SHOCCS_GEN_OPERATORS(operator-=, -=)
    SHOCCS_GEN_OPERATORS(operator*=, *=)
    SHOCCS_GEN_OPERATORS(operator/=, /=) 
};
#undef SHOCCS_GEN_OPERATORS

template <typename T>
struct ViewMath;

class ViewAccess
{
    template <typename T>
    friend class ViewMath;
};

#define SHOCCS_GEN_OPERATORS(op, f)                                                      \
    template <typename U, Numeric V>                                                     \
    requires std::derived_from<std::remove_cvref_t<U>, T> friend constexpr auto op(      \
        U&& u, V)                                                                        \
    {                                                                                    \
        return u;                                                                        \
    }                                                                                    \
                                                                                         \
    template <typename U, Numeric V>                                                     \
    requires std::derived_from<std::remove_cvref_t<U>, T> friend constexpr auto op(      \
        V, U&& u)                                                                        \
    {                                                                                    \
        return u;                                                                        \
    }                                                                                    \
    template <typename U, typename V>                                                    \
    requires std::derived_from<std::remove_cvref_t<U>, T> friend constexpr auto op(      \
        U&& u, V&&)                                                                      \
    {                                                                                    \
        return u;                                                                        \
    }

template <typename T>
struct ViewMath {

private:
    SHOCCS_GEN_OPERATORS(operator+, std::plus{})
    SHOCCS_GEN_OPERATORS(operator-, std::minus{})
    SHOCCS_GEN_OPERATORS(operator*, std::multiplies{})
    SHOCCS_GEN_OPERATORS(operator/, std::divides{})

#if 0
    template <typename U, typename V>
    requires std::derived_from<std::remove_cvref_t<U>, T> friend constexpr auto
    operator+(U&& u, V&& v)
    {
    }
#endif
};

#undef SHOCCS_GEN_OPERATORS
} // namespace ccs::field::lazy
