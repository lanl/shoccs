#pragma once

#include <range/v3/range/concepts.hpp>
#include <range/v3/view/repeat.hpp>
#include <range/v3/view/zip.hpp>
#include <range/v3/view/zip_with.hpp>

#include <concepts>
#include <functional>

#include "types.hpp"

namespace ccs
{

namespace vs = ranges::views;

template <ranges::input_range R>
struct result_range;

template <template <typename> typename C, typename T>
class result_field;

namespace detail
{
// define some traits and concepts to constrain our universal references and ranges
template <typename = void>
struct is_result_field : std::false_type {
};
template <template <typename> typename C, typename T>
struct is_result_field<result_field<C, T>> : std::true_type {
};

template <typename U>
constexpr bool is_result_field_v = is_result_field<std::remove_cvref_t<U>>::value;

template <typename = void>
struct is_result_range : std::false_type {
};
template <typename T>
struct is_result_range<result_range<T>> : std::true_type {
};

template <typename U>
constexpr bool is_result_range_v = is_result_range<std::remove_cvref_t<U>>::value;

template <typename U>
constexpr bool is_result_v = is_result_field_v<U> || is_result_range_v<U>;

} // namespace detail

template <typename T>
concept Result = detail::is_result_v<T>;

template <ranges::input_range R>
struct result_range {
    R r;

    const R& range() const& { return r; }
    R& range() & { return r; }
    R range() && { return std::move(r); }

    // iterator interface
    size_t size() const noexcept requires ranges::sized_range<R> { return r.size(); }

    auto begin() const { return r.begin(); }
    auto begin() { return r.begin(); }
    auto end() const { return r.end(); }
    auto end() { return r.end(); }

    decltype(auto) operator[](int i) & requires ranges::random_access_range<R>
    {
        return r[i];
    }
    auto operator[](int i) && requires ranges::random_access_range<R> { return r[i]; }
    const auto& operator[](int i) const& requires ranges::random_access_range<R>
    {
        return r[i];
    }
};

template <ranges::input_range R>
result_range(R r) -> result_range<R>;

template <template <typename> typename C, typename T>
class result_field
{
    using S = result_field<C, T>;

protected:
    C<T> f;

public:
    result_field() = default;

    result_field(const C<T>& f) : f{f} {}
    result_field(C<T>&& f) : f{std::move(f)} {}
    result_field(int sz) requires std::is_constructible_v<C<T>, int> : f(sz) {}

    // do not provide a constructor from an input range since C may be non-owning
    // and that would make such an operator dangerous

#define gen_operators(op, acc)                                                           \
    template <Result R>                                                                  \
    requires(!std::same_as<S, std::remove_cvref<R>>) result_field& op(R&& r)             \
    {                                                                                    \
        constexpr bool can_resize = requires(R a, C<T> b) { b.resize(a.size()); };       \
        if constexpr (can_resize) f.resize(r.size());                                    \
                                                                                         \
        int sz = f.size();                                                               \
        for (int i = 0; auto&& v : r) {                                                  \
            if (i == sz) break;                                                          \
            f[i] acc v;                                                                  \
            ++i;                                                                         \
        }                                                                                \
        return *this;                                                                    \
    }                                                                                    \
    template <Numeric N>                                                                 \
    result_field& op(N n)                                                                \
    {                                                                                    \
        for (auto&& v : f) v acc n;                                                      \
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

        T&
        operator[](int i)
    {
        return f[i];
    }
    const T& operator[](int i) const { return f[i]; }

    // allow several kinds of indexing for easy use
    const T& operator()(int i) const { return f[i]; };
    T& operator()(int i) { return f[i]; }

    // iterator interface
    size_t size() const noexcept { return f.size(); }

    auto begin() const { return f.begin(); }
    auto begin() { return f.begin(); }
    auto end() const { return f.end(); }
    auto end() { return f.end(); }

    // allow conversion to spans
    operator std::span<T>() { return f; }

    const C<T>& range() const& { return f; }
    C<T>& range() & { return f; }
    C<T> range() && { return std::move(f); }
};

#define ret_range(expr)                                                                  \
    return scalar_range<decltype(expr), I> { expr, t.extents() }

#define gen_operators(op, f)                                                             \
    template <Result T, Result U>                                                        \
    constexpr auto op(T&& t, U&& u)                                                      \
    {                                                                                    \
        return result_range{                                                             \
            vs::zip_with(f, std::forward<T>(t).range(), std::forward<U>(u).range())};    \
    }                                                                                    \
    template <Result T, Numeric U>                                                       \
    constexpr auto op(T&& t, U u)                                                        \
    {                                                                                    \
        return result_range{vs::zip_with(f, std::forward<T>(t).range(), vs::repeat(u))}; \
    }                                                                                    \
    template <Result T, Numeric U>                                                       \
    constexpr auto op(U u, T&& t)                                                        \
    {                                                                                    \
        return result_range{vs::zip_with(f, vs::repeat(u), std::forward<T>(t).range())}; \
    }

// clang-format off
gen_operators(operator+, std::plus{})

gen_operators(operator-, std::minus{})

gen_operators(operator*, std::multiplies{})

gen_operators(operator/, std::divides{})

#undef gen_operators
#undef ret_range
    // clang-format on

    template <typename T = real>
    using result_view_t = result_field<std::span, T>;
using result_view = result_view_t<real>;
} // namespace ccs