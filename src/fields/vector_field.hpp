#pragma once

#include "scalar_field.hpp"

#include <range/v3/view/take_exactly.hpp>

namespace ccs
{

template <typename T = real>
class vector_field;

template <typename X, typename Y, typename Z>
struct vector_range;

namespace traits
{
// define some traits and concepts to constrain our universal references and ranges
template <typename = void>
struct is_vector_field : std::false_type {
};
template <typename T>
struct is_vector_field<vector_field<T>> : std::true_type {
};

template <typename U>
constexpr bool is_vector_field_v = is_vector_field<std::remove_cvref_t<U>>::value;

template <typename = void>
struct is_vector_range : std::false_type {
};
template <typename X, typename Y, typename Z>
struct is_vector_range<vector_range<X, Y, Z>> : std::true_type {
};

template <typename U>
constexpr bool is_vector_range_v = is_vector_range<std::remove_cvref_t<U>>::value;

template <typename U>
constexpr bool is_vrange_or_vfield_v = is_vector_field_v<U> || is_vector_range_v<U>;

} // namespace traits

template <typename T>
concept Vector_Field = traits::is_vrange_or_vfield_v<T>;

template <typename X, typename Y = X, typename Z = Y>
struct vector_range {
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
};

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

    template <Scalar XX, Scalar YY, Scalar ZZ>
    vector_field(XX&& x, YY&& y, ZZ&& z)
        : x{std::forward<XX>(x)}, y{std::forward<YY>(y)}, z{std::forward<ZZ>(z)}
    {
    }

    template <Scalar A>
    vector_field(A&& a)
        : x{a}, y{a}, z{std::forward<A>(a)}
    {
    }

#define gen_operators(op, acc)                                                           \
    template <Scalar R>                                                                  \
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

    // clang-format off

gen_operators(operator=, =)
gen_operators(operator+=, +=)
gen_operators(operator-=, -=)
gen_operators(operator*=, *=)
gen_operators(operator/=, /=)
#undef gen_operators

        // clang-format on

        int3 extents() const 
    {
        return x.extents_;
    }

    int3 size() const { return {(int)x.size(), (int)y.size(), (int)z.size()}; }
};

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
} // namespace ccs