#pragma once

#include "types.hpp"

#include <concepts>
#include <functional>

#include <range/v3/algorithm/copy.hpp>
#include <range/v3/range/concepts.hpp>
#include <range/v3/view/repeat_n.hpp>
#include <range/v3/view/zip_with.hpp>

//#include "range_operators.hpp"

namespace ccs
{

namespace vs = ranges::views;

template <typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

template <typename T = real, int = 2>
class scalar_field;

template <typename R, int I>
struct scalar_range;

namespace detail
{
template <typename = void>
struct is_scalar_field : std::false_type {
};
template <typename T, int I>
struct is_scalar_field<scalar_field<T, I>> : std::true_type {
};

template <typename U>
constexpr bool is_scalar_field_v = is_scalar_field<std::remove_cvref_t<U>>::value;

template <typename = void>
struct is_scalar_range : std::false_type {
};
template <typename T, int I>
struct is_scalar_range<scalar_range<T, I>> : std::true_type {
};

template <typename U>
constexpr bool is_scalar_range_v = is_scalar_range<std::remove_cvref_t<U>>::value;

template <typename U>
constexpr bool is_range_or_field_v = is_scalar_field_v<U> || is_scalar_range_v<U>;

template <typename>
struct scalar_dim_;

template <typename T, int I>
struct scalar_dim_<scalar_range<T, I>> {
    static constexpr int dim = I;
};

template <typename T, int I>
struct scalar_dim_<scalar_field<T, I>> {
    static constexpr int dim = I;
};

template <typename T>
requires is_range_or_field_v<std::remove_cvref_t<T>> constexpr int scalar_dim =
    scalar_dim_<std::remove_cvref_t<T>>::dim;

template <typename T, typename U>
requires is_range_or_field_v<T>&& is_range_or_field_v<U> constexpr bool is_same_dim_v =
    scalar_dim<T> == scalar_dim<U>;

} // namespace detail

template <typename T>
concept Field = detail::is_range_or_field_v<T>;

template <typename T, typename U>
concept Compatible_Fields = Field<T>&& Field<U>&& detail::is_same_dim_v<T, U>;

template <typename R, int I>
struct scalar_range {
    R r;

    const R& range() const& { return r; }
    R& range() & { return r; }
    R range() && { return std::move(r); }

    // iterator interface
    size_t size() const noexcept { return r.size(); }

    auto begin() const { return r.begin(); }
    auto begin() { return r.begin(); }
    auto end() const { return r.end(); }
    auto end() { return r.end(); }
};

template <typename T, int I>
class scalar_field
{
    std::vector<T> f;

public:
    scalar_field() = default;

    scalar_field(int n) : f(n) {}

    scalar_field(std::vector<T>&& f) : f{std::move(f)} {}

    template <typename R>
    requires Compatible_Fields<scalar_field<T, I>, R> scalar_field(R&& r) : f(r.size())
    {
        ranges::copy(r, f.begin());
    }

    // define copy assigmnent if R is owning
    template <typename R>
    requires Compatible_Fields<scalar_field<T, I>, R> scalar_field& operator=(R&& r)
    {
        f.resize(r.size());
        ranges::copy(r, f.begin());
        return *this;
    }

    // iterator interface
    size_t size() const noexcept { return f.size(); }

    auto begin() const { return f.begin(); }
    auto begin() { return f.begin(); }
    auto end() const { return f.end(); }
    auto end() { return f.end(); }

    const std::vector<T>& range() const& { return f; }
    std::vector<T>& range() & { return f; }
    std::vector<T> range() && { return std::move(f); }
};

#define ret_range(expr)                                                                  \
    return scalar_range<decltype(expr), I> { expr }

#define gen_operators(op, f)                                                             \
    template <typename T, typename U>                                                    \
    requires Compatible_Fields<T, U> constexpr auto op(T&& t, U&& u)                     \
    {                                                                                    \
        constexpr auto I = detail::scalar_dim<T>;                                        \
        ret_range(                                                                       \
            vs::zip_with(f, std::forward<T>(t).range(), std::forward<U>(u).range()));    \
    }                                                                                    \
    template <Field T, Numeric U>                                                        \
    constexpr auto op(T&& t, U u)                                                        \
    {                                                                                    \
        constexpr auto I = detail::scalar_dim<T>;                                        \
        ret_range(                                                                       \
            vs::zip_with(f, std::forward<T>(t).range(), vs::repeat_n(u, t.size())));     \
    }                                                                                    \
    template <Field T, Numeric U>                                                        \
    constexpr auto op(U u, T&& t)                                                        \
    {                                                                                    \
        constexpr auto I = detail::scalar_dim<T>;                                        \
        ret_range(                                                                       \
            vs::zip_with(f, vs::repeat_n(u, t.size()), std::forward<T>(t).range()));     \
    }

gen_operators(operator+, std::plus{})

gen_operators(operator-, std::minus{})

gen_operators(operator*, std::multiplies{})

gen_operators(operator/, std::divides{})

#undef gen_operators
#undef ret_range

} // namespace ccs