#pragma once

#include "field_fwd.hpp"
#include "field_math.hpp"
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

template <typename T>
class field : field_math<field<T>>
{
    friend class field_math_access;

    std::vector<ccs::scalar<T>> scalars_;
    std::vector<ccs::vector<T>> vectors_;

public:
    field() = default;

    field(system_size sz)
    {
        auto&& [ns, nv, ss] = sz;
        scalars_.reserve(ns);
        vectors_.reserve(nv);

        for (integer i = 0; i < ns; i++) { scalars_.emplace_back(ss); }
        for (integer i = 0; i < nv; i++) { vectors_.emplace_back(tuple{ss, ss, ss}); }
    }

    field(std::vector<ccs::scalar<T>> ss, std::vector<ccs::vector<T>> vs)
        : scalars_{MOVE(ss)}, vectors_{MOVE(vs)}
    {
    }

    // Systemsystem_Field(int nfields, int3 extents, solid_points);

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

    std::span<ccs::scalar<T>> scalars() { return scalars_; }

    std::span<const ccs::scalar<T>> scalars() const { return scalars_; }

    std::span<ccs::vector<T>> vectors() { return vectors_; }

    std::span<const ccs::vector<T>> vectors() const { return vectors_; }

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
