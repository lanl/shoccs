#pragma once

#include "scalar.hpp"
#include "selector.hpp"
#include "vector.hpp"
#include <utility>

namespace ccs
{

struct system_size {
    integer nscalars;
    integer nvectors;
    scalar<integer> scalar_size;
};

template <typename T>
class system_field
// : lazy::view_math<system_field<T>>, lazy::container_math<system_field<T>>
{
    //     friend class lazy::view_access;
    //     friend class lazy::container_access;

    std::vector<ccs::scalar<T>> scalars_;
    std::vector<ccs::vector<T>> vectors_;

public:
    using type = system_field<T>;

    system_field() = default;

    system_field(system_size sz)
    {
        auto&& [ns, nv, ss] = sz;
        scalars_.reserve(ns);
        vectors_.reserve(nv);

        for (integer i = 0; i < ns; i++) { scalars_.emplace_back(ss); }
        for (integer i = 0; i < nv; i++) { vectors_.emplace_back(tuple{ss, ss, ss}); }
    }

    // Systemsystem_Field(int nsystem_fields, int3 extents, solid_points);

    // construction from an invocable
    template <std::invocable<type&> F>
    system_field(F&& f)
    {
        std::invoke(FWD(f), *this);
    }

    // assignment from an invocable of the same type
    template <std::invocable<type&> F>
    system_field& operator=(F&& f)
    {
        std::invoke(FWD(f), *this);
        return *this;
    }

    template <Numeric V>
    system_field& operator=(V&&)
    {
        return *this;
    }

    // conversion operator
    template <typename U>
        requires std::convertible_to<T&, U>
    operator system_field<U>() { return system_field<U>{}; }

    // api for resizing should only be available for true containers
    int nscalars() const { return scalars_.size(); }

    system_field& nscalars(int n)
    {
        scalars_.resize(n);
        return *this;
    }

    int nvectors() const { return vectors_.size(); }

    system_field& nvectors(int n)
    {
        vectors_.resize(n);
        return *this;
    }

    int3 extents() const { return {}; }

    system_field& extents(int3) { return *this; }

    const auto& solid() { return scalars_; }

    system_field& solid(int) { return *this; }

    system_field& object_boundaries(int3) { return *this; }

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

    constexpr void swap(system_field& other) noexcept
    {
        using std::swap;
        swap(this->scalars_, other.scalars_);
        swap(this->vectors_, other.vectors_);
    }

    friend constexpr void swap(system_field& a, system_field& b) noexcept { a.swap(b); }
};

using field = system_field<std::vector<real>>;
using field_span = system_field<std::span<real>>;
using field_view = system_field<std::span<const real>>;

} // namespace ccs
