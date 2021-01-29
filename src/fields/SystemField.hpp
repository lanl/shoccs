#pragma once

#include "lazy_field_math.hpp"
#include "r_tuple.hpp"
#include <utility>

namespace ccs
{
namespace field
{

template <typename T>
class SystemField : lazy::ViewMath<SystemField<T>>, lazy::ContainerMath<SystemField<T>>
{
    friend class lazy::ViewAccess;
    friend class lazy::ContainerAccess;

    std::vector<T> fields;

public:
    using type = SystemField<T>;

    SystemField() = default;

    // SystemField(int nfields, int3 extents, solid_points);

    // construction from an invocable
    template <std::invocable<type&> F>
    SystemField(F&& f)
    {
        std::invoke(FWD(f), *this);
    }

    // assignment from an invocable of the same type
    template <std::invocable<type&> F>
    SystemField& operator=(F&& f)
    {
        std::invoke(FWD(f), *this);
        return *this;
    }

    template <Numeric V>
    SystemField& operator=(V&&)
    {
        return *this;
    }

    // conversion operator
    template <typename U>
    requires std::convertible_to<T&, U> operator SystemField<U>()
    {
        return SystemField<U>{};
    }

    // api for resizing should only be available for true containers
    int nfields() const { return fields.size(); }

    SystemField& nfields(int n)
    {
        fields.resize(n);
        return *this;
    }

    int3 extents() const { return {}; }

    SystemField& extents(int3) { return *this; }

    const auto& solid() { return fields; }

    SystemField& solid(int) { return *this; }

    SystemField& object_boundaries(int3) { return *this; }

    template <std::integral... Is>
    auto operator()(Is&&... i) const
    {
        // use `forward_as_tuple` to get a tuple of references
        return std::forward_as_tuple(fields[i]...);
    }

    template <typename... Is>
    requires(std::is_enum_v<Is>&&...) auto operator()(Is&&... i) const
    {
        return (*this)(
            static_cast<std::underlying_type_t<std::remove_cvref_t<Is>>>(FWD(i))...);
    }

    template <std::integral... Is>
    auto operator()(Is&&... i)
    {
        return std::forward_as_tuple(fields[i]...);
    }

    template <typename... Is>
    requires(std::is_enum_v<Is>&&...) auto operator()(Is&&... i)
    {
        return (*this)(
            static_cast<std::underlying_type_t<std::remove_cvref_t<Is>>>(FWD(i))...);
    }

    constexpr void swap(SystemField& other) noexcept
    {
        using std::swap;
        swap(this->fields, other.fields);
    }

    friend constexpr void swap(SystemField& a, SystemField& b) noexcept { a.swap(b); }
};
} // namespace field

using SystemField = field::SystemField<std::vector<real>>;
using SystemView_Mutable = field::SystemField<std::span<real>>;
using SystemView_Const = field::SystemField<std::span<const real>>;

} // namespace ccs