#pragma once

#include <range/v3/core.hpp>
#include <range/v3/range/concepts.hpp>
#include <range/v3/view/repeat.hpp>
#include <range/v3/view/repeat_n.hpp>
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
struct result_range : R {
#if 0
private:
    friend ranges::range_access;

    R r;

    struct cursor {
    private:
        using It = ranges::iterator_t<R>;
        It iter;

    public:
        // using contiguous = ;

        cursor() = default;
        cursor(It it) : iter(it) {}

        decltype(auto) read() const { return *iter; }
        decltype(auto) read() { return *iter; }

        bool equal(const cursor& other) const { return iter == other.iter; }

        void next() { ++iter; }

        void prev() requires ranges::bidirectional_iterator<It> { --iter; }

        auto distance_to(const cursor& other) const
            -> decltype(ranges::distance(iter, other.iter))
        {
            return ranges::distance(iter, other.iter);
        }

        auto advance(std::ptrdiff_t n) -> decltype(ranges::advance(iter, n))
        {
            ranges::advance(iter, n);
        }
    };

    cursor begin_cursor() const { return {r.begin()}; }
    cursor begin_cursor() { return {r.begin()}; }

    cursor end_cursor() const { return {ranges::end(r)}; }
    cursor end_cursor() { return {ranges::end(r)}; }
#endif
public:
    result_range() = default;

    result_range(R&& r) : R{std::forward<R>(r)} {}
};

template <typename R>
result_range<R> result(R&& r)
{
    return {std::forward<R>(r)};
}

template <template <typename> typename C, typename T>
class result_field : public ranges::view_facade<result_field<C, T>>
{
    using S = result_field<C, T>;
    using F = C<T>;
    friend ranges::range_access;

protected:
    F f;

private:
    struct cursor {
    private:
        using It = ranges::iterator_t<F>;
        It iter;

    public:
        using contiguous = std::true_type;

        cursor() = default;
        cursor(It it) : iter(it) {}

        decltype(auto) read() const { return *iter; }
        decltype(auto) read() { return *iter; }

        bool equal(const cursor& other) const { return iter == other.iter; }

        void next() { ++iter; }

        void prev() { --iter; }

        auto distance_to(const cursor& other) const
            -> decltype(ranges::distance(iter, other.iter))
        {
            return ranges::distance(iter, other.iter);
        }

        auto advance(std::ptrdiff_t n) -> decltype(ranges::advance(iter, n))
        {
            ranges::advance(iter, n);
        }
    };

    cursor begin_cursor() const { return {ranges::begin(f)}; }
    cursor begin_cursor() { return {ranges::begin(f)}; }

    cursor end_cursor() const { return {ranges::end(f)}; }
    cursor end_cursor() { return {ranges::end(f)}; }

public:
    result_field() = default;

    result_field(const F& f) : f{f} {}
    result_field(F&& f) : f{std::move(f)} {}
    result_field(int sz) requires std::is_constructible_v<F, int> : f(sz) {}

    // do not provide a constructor from an input range since C may be non-owning
    // and that would make such an operator dangerous

#define gen_operators(op, acc)                                                           \
    template <Result R>                                                                  \
    requires(!std::same_as<S, std::remove_cvref<R>>) result_field& op(R&& r)             \
    {                                                                                    \
        constexpr bool can_resize = requires(R a, F b) { b.resize(a.size()); };          \
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

#if 0
        T&
        operator[](int i)
    {
        return f[i];
    }
    const T& operator[](int i) const { return f[i]; }
#endif
        // allow several kinds of indexing for easy use
        const T&
        operator()(int i) const
    {
        return f[i];
    };
    T& operator()(int i) { return f[i]; }

    // allow conversion to spans
    operator std::span<T>() { return f; }

#if 0
    const F& range() const& { return f; }
    F& range() & { return f; }
    F range() && { return std::move(f); }
#endif
}; // namespace ccs

#define gen_operators(op, f)                                                             \
    template <Result T, Result U>                                                        \
    constexpr auto op(T&& t, U&& u)                                                      \
    {                                                                                    \
        return result(vs::zip_with(f, std::forward<T>(t), std::forward<U>(u)));          \
    }                                                                                    \
    template <Result T, Numeric U>                                                       \
    constexpr auto op(T&& t, U u)                                                        \
    {                                                                                    \
        if constexpr (ranges::sized_range<T>)                                            \
            return result(                                                               \
                vs::zip_with(f, std::forward<T>(t), vs::repeat_n(u, t.size())));         \
        else                                                                             \
            return result(vs::zip_with(f, std::forward<T>(t), vs::repeat(u)));           \
    }                                                                                    \
    template <Result T, Numeric U>                                                       \
    constexpr auto op(U u, T&& t)                                                        \
    {                                                                                    \
        if constexpr (ranges::sized_range<T>)                                            \
            return result(                                                               \
                vs::zip_with(f, vs::repeat_n(u, t.size()), std::forward<T>(t)));         \
        else                                                                             \
            return result(vs::zip_with(f, vs::repeat(u), std::forward<T>(t)));           \
    }

// clang-format off
gen_operators(operator+, std::plus{})

gen_operators(operator-, std::minus{})

gen_operators(operator*, std::multiplies{})

gen_operators(operator/, std::divides{})

#undef gen_operators

    // clang-format on

    template <typename T = real>
    using result_view_t = result_field<std::span, T>;
using result_view = result_view_t<real>;

} // namespace ccs