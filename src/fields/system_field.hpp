#pragma once

#include "lazy_field_math.hpp"
#include "scalar.hpp"
#include "vector.hpp"
#include <utility>

namespace ccs
{
namespace detail
{

template <typename T>
class field : lazy::view_math<field<T>>, lazy::container_math<field<T>>
{
    friend class lazy::view_access;
    friend class lazy::container_access;

    std::vector<ccs::scalar<T>> scalars_;
    std::vector<ccs::vector<T>> vectors_;

public:
    using type = field<T>;

    field() = default;

    // SystemField(int nfields, int3 extents, solid_points);

    // construction from an invocable
    template <std::invocable<type&> F>
    field(F&& f)
    {
        std::invoke(FWD(f), *this);
    }

    // assignment from an invocable of the same type
    template <std::invocable<type&> F>
    field& operator=(F&& f)
    {
        std::invoke(FWD(f), *this);
        return *this;
    }

    template <Numeric V>
    field& operator=(V&&)
    {
        return *this;
    }

    // conversion operator
    template <typename U>
        requires std::convertible_to<T&, U>
    operator field<U>() { return field<U>{}; }

    // api for resizing should only be available for true containers
    int nscalars() const { return scalars_.size(); }

    field& nscalars(int n)
    {
        scalars_.resize(n);
        return *this;
    }

    int nvectors() const { return vectors_.size(); }

    field& nvectors(int n)
    {
        vectors_.resize(n);
        return *this;
    }

    int3 extents() const { return {}; }

    field& extents(int3) { return *this; }

    const auto& solid() { return scalars_; }

    field& solid(int) { return *this; }

    field& object_boundaries(int3) { return *this; }

    template <std::integral... Is>
    auto scalars(Is&&... i) const
    {
        // use `forward_as_tuple` to get a tuple of references
        return std::forward_as_tuple(scalars_[i]...);
    }

    template <typename... Is>
        requires(std::is_enum_v<Is>&&...)
    auto scalars(Is&&... i) const
    {
        return scalars(
            static_cast<std::underlying_type_t<std::remove_cvref_t<Is>>>(FWD(i))...);
    }

    template <std::integral... Is>
    auto scalars(Is&&... i)
    {
        return std::forward_as_tuple(scalars_[i]...);
    }

    template <typename... Is>
        requires(std::is_enum_v<Is>&&...)
    auto scalars(Is&&... i)
    {
        return scalars(
            static_cast<std::underlying_type_t<std::remove_cvref_t<Is>>>(FWD(i))...);
    }

    constexpr void swap(field& other) noexcept
    {
        using std::swap;
        swap(this->scalars_, other.scalars_);
        swap(this->vectors_, other.vectors_);
    }

    friend constexpr void swap(field& a, field& b) noexcept { a.swap(b); }
};

} // namespace detail

using field = detail::field<std::vector<real>>;
using field_span = detail::field<std::span<real>>;
using field_view = detail::field<std::span<const real>>;

} // namespace ccs
