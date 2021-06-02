#pragma once

#include "field_fwd.hpp"
#include "field_math.hpp"
#include <range/v3/algorithm/swap_ranges.hpp>
#include <utility>

namespace ccs
{

struct system_size {
    integer nscalars;
    integer nvectors;
    scalar<integer> scalar_size;
};

namespace detail
{

template <Range S, Range V>
class field : field_math<field<S, V>>
{
    friend class field_math_access;

    S s;
    V v;

public:
    field() = default;

    field(system_size sz) requires requires(S s, V v, ccs::scalar<integer> ss)
    {
        s.emplace_back(ss);
        v.emplace_back(tuple{ss, ss, ss});
    }
    {
        auto&& [ns, nv, ss] = sz;
        // reserve if we can
        if constexpr (requires(S s, V v, integer n) {
                          s.reserve(n);
                          v.reserve(n);
                      }) {
            s.reserve(ns);
            v.reserve(nv);
        }

        for (integer i = 0; i < ns; i++) { s.emplace_back(ss); }
        for (integer i = 0; i < nv; i++) { v.emplace_back(tuple{ss, ss, ss}); }
    }

    constexpr field(S&& s, V&& v) : s{FWD(s)}, v{FWD(v)} {}

    template <Field F>
        requires(ConstructibleFromRange<S, scalar_type<F>>&&
                     ConstructibleFromRange<V, vector_type<F>>)
    constexpr field(F&& f)
        : s(rs::begin(f.scalars()), rs::end(f.scalars())),
          v(rs::begin(f.vectors()), rs::end(f.vectors()))
    {
    }

    // construction from an invocable
    template <std::invocable<field&> F>
    field(F&& f)
    {
        std::invoke(FWD(f), *this);
    }

    // assignment from an invocable of the same type
    template <std::invocable<field&> F>
    field& operator=(F&& f)
    {
        std::invoke(FWD(f), *this);
        return *this;
    }

    template <Field F>
    requires std::is_assignable_v<rs::range_reference_t<S>, scalar_ref_t<F>> &&
        std::is_assignable_v<rs::range_reference_t<V>, vector_ref_t<F>>
            field& operator=(F&& f)
    {
        for_each([](auto& u, auto&& v) { u = v; }, *this, FWD(f));
        return *this;
    }

    template <typename F>
        requires(!Field<F> && std::is_assignable_v<rs::range_reference_t<S>, F> &&
                 std::is_assignable_v<rs::range_reference_t<V>, F>)
    field& operator=(F&& f)
    {
        for_each([&f](auto& u) { u = f; }, *this);
        return *this;
    }

    // api for resizing should only be available for true containers
    constexpr int nscalars() const requires rs::sized_range<S> { return rs::size(s); }

    constexpr int nvectors() const requires rs::sized_range<V> { return rs::size(v); }

    constexpr auto& scalars() { return s; }

    constexpr const auto& scalars() const { return s; }

    constexpr auto& vectors() { return v; }

    constexpr const auto& vectors() const { return v; }

    template <std::integral... Is>
    decltype(auto) scalars(Is&&... i) const requires rs::random_access_range<S>
    {
        if constexpr (sizeof...(Is) > 1)
            return tuple(s[i]...);
        else
            return (s[i], ...);
    }

    template <typename... Is>
        requires(std::is_enum_v<Is>&&...)
    decltype(auto) scalars(Is&&... i) const
    {
        return scalars(
            static_cast<std::underlying_type_t<std::remove_cvref_t<Is>>>(FWD(i))...);
    }

    template <std::integral... Is>
    decltype(auto) scalars(Is&&... i) requires rs::random_access_range<S>
    {
        if constexpr (sizeof...(Is) > 1)
            return tuple{s[i]...};
        else
            return (s[i], ...);
    }

    template <typename... Is>
        requires(std::is_enum_v<Is>&&...)
    decltype(auto) scalars(Is&&... i)
    {
        return scalars(
            static_cast<std::underlying_type_t<std::remove_cvref_t<Is>>>(FWD(i))...);
    }

    constexpr void swap(field& other) noexcept requires requires(S s0, S s1, V v0, V v1)
    {
        rs::swap_ranges(s0, s1);
        rs::swap_ranges(v0, v1);
    }
    {
        rs::swap_ranges(s, other.s);
        rs::swap_ranges(v, other.v);
    }

    friend constexpr void swap(field& a, field& b) noexcept { a.swap(b); }
};

template <Range S, Range V>
field(S&&, V&&) -> field<viewable_range_by_value<S>, viewable_range_by_value<V>>;
} // namespace detail

using field = detail::field<std::vector<scalar_real>, std::vector<vector_real>>;
using field_span = detail::field<std::vector<scalar_span>, std::vector<vector_span>>;
using field_view = detail::field<std::vector<scalar_view>, std::vector<vector_view>>;

} // namespace ccs
