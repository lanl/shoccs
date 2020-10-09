#pragma once

#include "scalar_field.hpp"
#include "vector_field_fwd.hpp"

#include <range/v3/view/take_exactly.hpp>

namespace ccs
{

template <typename X_, typename Y_ = X_, typename Z_ = Y_>
struct vector_range {
    using X = X_;
    using Y = Y_;
    using Z = Z_;
    X x;
    Y y;
    Z z;

    int3 size() const requires requires(X x, Y y, Z z)
    {
        x.size();
        y.size();
        z.size();
    }
    {
        return int3{(int)x.size(), (int)y.size(), (int)z.size()};
    }

    void resize(int n) requires requires(X x, Y y, Z z, int i)
    {
        x.resize(i);
        y.resize(i);
        z.resize(i);
    }
    {
        x.resize(n);
        y.resize(n);
        z.resize(n);
    }

    void resize(int3 n) requires requires(X x, Y y, Z z, int i)
    {
        x.resize(i);
        y.resize(i);
        z.resize(i);
    }
    {
        x.resize(n[0]);
        y.resize(n[1]);
        z.resize(n[2]);
    }

    int3 extents() const requires requires(X x) { x.extents(); }
    {
        return x.extents();
    }

#define SHOCCS_GEN_OPERATORS_(op, acc)                                                   \
    template <Numeric N>                                                                 \
    vector_range& op(N n) requires rs::output_range<X, N>&& rs::output_range<Y, N>&&     \
        rs::output_range<Z, N>                                                           \
    {                                                                                    \
        for (auto&& v : x) v acc n;                                                      \
        for (auto&& v : y) v acc n;                                                      \
        for (auto&& v : z) v acc n;                                                      \
        return *this;                                                                    \
    }

    SHOCCS_GEN_OPERATORS_(operator=, =)

#define SHOCCS_GEN_OPERATORS(op, acc)                                                    \
    SHOCCS_GEN_OPERATORS_(op, acc)                                                       \
    template <Vector_Field F>                                                            \
    vector_range& op(F&& f)                                                              \
    {                                                                                    \
        for (auto&& [v, n] : vs::zip(x, f.x)) v acc n;                                   \
        for (auto&& [v, n] : vs::zip(y, f.y)) v acc n;                                   \
        for (auto&& [v, n] : vs::zip(z, f.z)) v acc n;                                   \
        return *this;                                                                    \
    }

    SHOCCS_GEN_OPERATORS(operator+=, +=)
    SHOCCS_GEN_OPERATORS(operator-=, -=)
    SHOCCS_GEN_OPERATORS(operator*=, *=)
    SHOCCS_GEN_OPERATORS(operator/=, /=)

#undef SHOCCS_GEN_OPERATORS
#undef SHOCCS_GEN_OPERATORS_
}; // namespace ccs

template <typename X, typename Y, typename Z>
vector_range(X&&, Y&&, Z &&) -> vector_range<X, Y, Z>;

template <typename T>
struct vector_field {
    using X = scalar_field<T, 0>;
    using Y = scalar_field<T, 1>;
    using Z = scalar_field<T, 2>;

    X x;
    Y y;
    Z z;

    vector_field() = default;

    vector_field(int3 ex) : x{ex}, y{ex}, z{ex} {}

    template <Scalar_Field XX, Scalar_Field YY, Scalar_Field ZZ>
    vector_field(XX&& x, YY&& y, ZZ&& z)
        : x{std::forward<XX>(x)}, y{std::forward<YY>(y)}, z{std::forward<ZZ>(z)}
    {
    }

    template <Scalar_Field A>
    vector_field(A&& a) : x{a}, y{a}, z{std::forward<A>(a)}
    {
    }

#define SHOCCS_GEN_OPERATORS_(op, acc)                                                   \
    template <Scalar_Field R>                                                            \
    vector_field& op(R&& r)                                                              \
    {                                                                                    \
        x acc r;                                                                         \
        y acc r;                                                                         \
        z acc r;                                                                         \
                                                                                         \
        return *this;                                                                    \
    }                                                                                    \
    template <Numeric N>                                                                 \
    vector_field& op(N n)                                                                \
    {                                                                                    \
        x acc n;                                                                         \
        y acc n;                                                                         \
        z acc n;                                                                         \
        return *this;                                                                    \
    }

    SHOCCS_GEN_OPERATORS_(operator=, =)

#define SHOCCS_GEN_OPERATORS(op, acc)                                                    \
    SHOCCS_GEN_OPERATORS_(op, acc)                                                       \
                                                                                         \
    template <Vector_Field R>                                                            \
    vector_field& op(R&& r)                                                              \
    {                                                                                    \
        x acc r.x;                                                                       \
        y acc r.y;                                                                       \
        z acc r.z;                                                                       \
        return *this;                                                                    \
    }

    SHOCCS_GEN_OPERATORS(operator+=, +=)
    SHOCCS_GEN_OPERATORS(operator-=, -=)
    SHOCCS_GEN_OPERATORS(operator*=, *=)
    SHOCCS_GEN_OPERATORS(operator/=, /=)

#undef SHOCCS_GEN_OPERATORS
#undef SHOCCS_GEN_OPERATORS_

    int3 extents() const { return x.extents_; }

    int3 size() const { return {(int)x.size(), (int)y.size(), (int)z.size()}; }
};

// define math operations on vector fields if they are defined on their componenents
#define SHOCCS_GEN_OPERATORS(op, f)                                                      \
    template <Vector_Field T, Vector_Field U>                                            \
    requires requires(T t, U u)                                                          \
    {                                                                                    \
        f(t.x, u.x);                                                                     \
        f(t.y, u.y);                                                                     \
        f(t.z, u.z);                                                                     \
    }                                                                                    \
    constexpr auto op(T&& t, U&& u)                                                      \
    {                                                                                    \
        return vector_range{f(t.x, u.x), f(t.y, u.y), f(t.z, u.z)};                      \
    }                                                                                    \
    template <Vector_Field T, Numeric U>                                                 \
    constexpr auto op(T&& t, U u)                                                        \
    {                                                                                    \
        constexpr bool direct_apply = requires(T t, U u)                                 \
        {                                                                                \
            f(t.x, u);                                                                   \
            f(t.y, u);                                                                   \
            f(t.z, u);                                                                   \
        };                                                                               \
        if constexpr (direct_apply)                                                      \
            return vector_range{f(t.x, u), f(t.y, u), f(t.z, u)};                        \
        else {                                                                           \
            const auto [nx, ny, nz] = t.size();                                          \
            return vector_range{                                                         \
                vs::zip_with(f, std::forward<T>(t).x, vs::repeat_n(u, nx)),              \
                vs::zip_with(f, std::forward<T>(t).y, vs::repeat_n(u, ny)),              \
                vs::zip_with(f, std::forward<T>(t).z, vs::repeat_n(u, nz))};             \
        }                                                                                \
    }                                                                                    \
    template <Vector_Field T, Numeric U>                                                 \
    constexpr auto op(U u, T&& t)                                                        \
    {                                                                                    \
        constexpr bool direct_apply = requires(T t, U u)                                 \
        {                                                                                \
            f(u, t.x);                                                                   \
            f(u, t.y);                                                                   \
            f(u, t.z);                                                                   \
        };                                                                               \
        if constexpr (direct_apply)                                                      \
            return vector_range{f(u, t.x), f(u, t.y), f(u, t.z)};                        \
        else {                                                                           \
            const auto [nx, ny, nz] = t.size();                                          \
            return vector_range{                                                         \
                vs::zip_with(f, vs::repeat_n(u, nx), std::forward<T>(t).x),              \
                vs::zip_with(f, vs::repeat_n(u, ny), std::forward<T>(t).y),              \
                vs::zip_with(f, vs::repeat_n(u, nz), std::forward<T>(t).z)};             \
        }                                                                                \
    }

SHOCCS_GEN_OPERATORS(operator+, std::plus{})
SHOCCS_GEN_OPERATORS(operator*, std::multiplies{})
SHOCCS_GEN_OPERATORS(operator-, std::minus{})
SHOCCS_GEN_OPERATORS(operator/, std::divides{})
#undef SHOCCS_GEN_OPERATORS

namespace detail
{
template <typename F, typename V>
concept Vector_Invocable = requires(V v, F f)
{
    f(v.x);
    f(v.y);
    f(v.z);
};

template <typename F, typename V>
concept Vector_View_Closure =
    Vector_Invocable<F, V>&& rs::invocable_view_closure<V, typename F::X>&&
        rs::invocable_view_closure<V, typename F::Y>&&
            rs::invocable_view_closure<V, typename F::Z>;
} // namespace detail

template <Vector_Field F, detail::Vector_Invocable<F> T>
constexpr auto operator>>(F&& f, T t)
{
    return vector_range{t(f.x), t(f.y), t(f.z)};
}

template <Vector_Field F, detail::Vector_View_Closure<F> ViewFn>
constexpr auto operator>>(F&& f, vs::view_closure<ViewFn> t)
{
    constexpr bool can_right_shift = requires(F f, vs::view_closure<ViewFn> t)
    {
        f.x >> t;
        f.y >> t;
        f.z >> t;
    };
    if constexpr (can_right_shift)
        return vector_range{f.x >> t, f.y >> t, f.z >> t};
    else
        return vector_range{f.x | t, f.y | t, f.z | t};
}

template <Vector_Field F, Vector_Field T>
constexpr auto operator>>(F&& f, T t)
{
    constexpr bool can_right_shift = requires(F f, T t)
    {
        f.x >> t.x;
        f.y >> t.y;
        f.z >> t.y;
    };
    if constexpr (can_right_shift)
        return vector_range{f.x >> t.x, f.y >> t.x, f.z >> t.y};
    else
        return vector_range{f.x | t.x, f.y | t.y, f.z | t.z};
}

constexpr auto vector_take(int3 sz)
{
    return vector_range{
        vs::take_exactly(sz[0]), vs::take_exactly(sz[1]), vs::take_exactly(sz[2])};
}

using v_field = vector_field<real>;
// template<typename... Args>
// using v_range = vector_range<Args...>;
// template<typename... Args>
// using v_tuple = vector_range<Args...>;
} // namespace ccs