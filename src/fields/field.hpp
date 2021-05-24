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

    template <Numeric N>
    field& operator=(N&&)
    {
        return *this;
    }

    // conversion operator
    // template <typename U>
    //     requires std::convertible_to<T&, U>
    // operator field<U>() { return field<U>{}; }

    // api for resizing should only be available for true containers
    constexpr int nscalars() const requires rs::sized_range<S> { return rs::size(s); }

    // field& nscalars(int n)
    // {
    //     scalars_.resize(n);
    //     return *this;
    // }

    constexpr int nvectors() const requires rs::sized_range<V> { return rs::size(v); }

    // field& nvectors(int n)
    // {
    //     vectors_.resize(n);
    //     return *this;
    // }

    // int3 extents() const { return {}; }

    // field& extents(int3) { return *this; }

    // const auto& solid() { return scalars_; }

    // field& solid(int) { return *this; }

    // field& object_boundaries(int3) { return *this; }

    constexpr auto scalars() { return vs::all(s); }

    constexpr auto scalars() const { return vs::all(s); }

    constexpr auto vectors() { return vs::all(v); }

    constexpr auto vectors() const { return vs::all(v); }

    template <std::integral... Is>
    auto scalars(Is&&... i) const requires rs::random_access_range<S>
    {
        return tuple(s[i]...);
    }

    template <typename... Is>
        requires(std::is_enum_v<Is>&&...)
    auto scalars(Is&&... i) const
    {
        return scalars(
            static_cast<std::underlying_type_t<std::remove_cvref_t<Is>>>(FWD(i))...);
    }

    template <std::integral... Is>
    auto scalars(Is&&... i) requires rs::random_access_range<S>
    {
        return tuple{s[i]...};
    }

    template <typename... Is>
        requires(std::is_enum_v<Is>&&...)
    auto scalars(Is&&... i)
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
using field_span = detail::field<std::span<scalar_real>, std::span<vector_real>>;
using field_view =
    detail::field<std::span<const scalar_real>, std::span<const vector_real>>;

} // namespace ccs
