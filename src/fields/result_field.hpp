#pragma once

#include <range/v3/algorithm/copy.hpp>
#include <range/v3/core.hpp>
#include <range/v3/range/concepts.hpp>
#include <range/v3/view/repeat.hpp>
#include <range/v3/view/repeat_n.hpp>
#include <range/v3/view/zip.hpp>
#include <range/v3/view/zip_with.hpp>

#include <concepts>
#include <functional>
#include <meta/meta.hpp>

#include "types.hpp"

namespace ccs
{

template <rs::range R>
struct result_range;

template <template <typename> typename C, typename T>
class result_field;

namespace traits
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

} // namespace traits

template <typename T>
concept Result = traits::is_result_v<T>;

// a simple wrapper around a range to avoid overloading math operators for all
// ranges. In addition to (naturally) supporting `|` via inheritance, we also
// use the `>>` operator to "lift" view functions so that the return value is a
// result_range rather than "just" a range-v3 view
template <rs::range R>
struct result_range : R {

    // allow several kinds of indexing for easy use
    decltype(auto) operator()(int i) const requires rs::random_access_range<R>
    {
        return (*this)[i];
    };
    decltype(auto) operator()(int i) requires rs::random_access_range<R>
    {
        return (*this)[i];
    }
};
template <typename R>
result_range(R &&) -> result_range<R>;

template <template <typename> typename C, typename T>
class result_field : public rs::view_facade<result_field<C, T>>
{
    using S = result_field<C, T>;
    using F = C<T>;
    friend rs::range_access;
    F f;

    template <bool Const>
    struct cursor {
    private:
        using R = meta::const_if_c<Const, F>;
        using It = rs::iterator_t<R>;
        It it;

    public:
        using contiguous = std::true_type;
        // using value_type = T;

        cursor() = default;
        cursor(It it) : it{it} {}

        decltype(auto) read() const { return *it; }
        decltype(auto) read() { return *it; }

        bool equal(const cursor& c) const { return it == c.it; }

        void next() { ++it; }
        void prev() { --it; }
        auto distance_to(const cursor& c) const { return c.it - it; }
        void advance(std::ptrdiff_t n) { it += n; }
    };

    cursor<true> begin_cursor() const { return {f.begin()}; }
    cursor<false> begin_cursor() { return {f.begin()}; }

    cursor<true> end_cursor() const { return {f.end()}; }
    cursor<false> end_cursor() { return {f.end()}; }

public:
    result_field() = default;

    result_field(const F& f) : f{f} {}
    result_field(F&& f) : f{MOVE(f)} {}
    // result_field(int sz) requires std::is_constructible_v<F, int> : f(sz) {}

    template <typename U>
        result_field(U&& u) requires std::constructible_from<F, U> &&
        (!std::same_as<F, std::remove_cvref_t<U>>) : f(FWD(u))
    {
    }

    auto resize(int sz) requires requires(F f, int sz) { f.resize(sz); }
    {
        f.resize(sz);
    }

    template <typename R>
        requires rs::input_range<R>&& rs::sized_range<R> &&
        (!std::same_as<S, std::remove_cvref_t<R>>)&&std::
            is_constructible_v<F, int> result_field(R&& r)
        : f(ranges::size(r))
    {
        ranges::copy(r, ranges::begin(f));
    }

#define SHOCCS_GEN_OPERATORS(op, acc)                                                    \
    template <typename R>                                                                \
        requires rs::input_range<R> &&                                                   \
        (!std::same_as<S, std::remove_cvref_t<R>>)result_field& op(R&& r)                \
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

    SHOCCS_GEN_OPERATORS(operator=, =)
    SHOCCS_GEN_OPERATORS(operator+=, +=)
    SHOCCS_GEN_OPERATORS(operator-=, -=)
    SHOCCS_GEN_OPERATORS(operator*=, *=)
    SHOCCS_GEN_OPERATORS(operator/=, /=)
#undef SHOCCS_GEN_OPERATORS

    // allow several kinds of indexing for easy use
    const T& operator()(int i) const { return f[i]; };
    T& operator()(int i) { return f[i]; }

    // allow conversion to spans
    // operator std::span<T>() { return f; }
};

#define SHOCCS_GEN_OPERATORS(op, f)                                                      \
    template <Result T, Result U>                                                        \
    constexpr auto op(T&& t, U&& u)                                                      \
    {                                                                                    \
        return result_range{vs::zip_with(f, FWD(t), FWD(u))};                            \
    }                                                                                    \
    template <Result T, Numeric U>                                                       \
    constexpr auto op(T&& t, U u)                                                        \
    {                                                                                    \
        if constexpr (rs::sized_range<T>)                                                \
            return result_range{vs::zip_with(f, FWD(t), vs::repeat_n(u, t.size()))};     \
        else                                                                             \
            return result_range{vs::zip_with(f, FWD(t), vs::repeat(u))};                 \
    }                                                                                    \
    template <Result T, Numeric U>                                                       \
    constexpr auto op(U u, T&& t)                                                        \
    {                                                                                    \
        if constexpr (ranges::sized_range<T>)                                            \
            return result_range{vs::zip_with(f, vs::repeat_n(u, t.size()), FWD(t))};     \
        else                                                                             \
            return result_range{vs::zip_with(f, vs::repeat(u), FWD(t))};                 \
    }

SHOCCS_GEN_OPERATORS(operator+, std::plus{})
SHOCCS_GEN_OPERATORS(operator-, std::minus{})
SHOCCS_GEN_OPERATORS(operator*, std::multiplies{})
SHOCCS_GEN_OPERATORS(operator/, std::divides{})
#undef SHOCCS_GEN_OPERATORS

template <Result R, typename ViewFn>
requires rs::invocable_view_closure<ViewFn, R> constexpr auto
operator>>(R&& r, vs::view_closure<ViewFn> t)
{
    return result_range{FWD(r) | t};
}

template <typename T = real>
using result_view_t = result_field<std::span, T>;
using result_view = result_view_t<real>;
} // namespace ccs