#pragma once

#include "scalar_field_fwd.hpp"
#include "vector_field_fwd.hpp"

namespace ccs
{

namespace detail
{
template <typename F, typename V>
concept Vector_Aggregrate = requires(V v, F f, int3 ijk)
{
    f(f(v.x(ijk), v.y(ijk)), v.z(ijk));
};

template <typename V>
concept Vector_of_Scalars =
    Vector_Field<V>&& Scalar<typename std::remove_cvref_t<V>::X>&& Scalar<
        typename std::remove_cvref_t<V>::Y>&& Scalar<typename std::remove_cvref_t<V>::Z>;
} // namespace detail

// forward decl public api
template <detail::Vector_of_Scalars V, detail::Vector_Aggregrate<V> F>
constexpr auto contract(V&& v, F&& f);

namespace detail
{
template <Vector_Field V, typename F>
struct contraction {
    V v;
    F f;

    auto size() const { return v.size()[0]; }
    auto extents() const { return v.extents(); }
    auto operator()(const int3& ijk) const { return f(f(v.x(ijk), v.y(ijk)), v.z(ijk)); }
};

template <Vector_Field V, typename F>
contraction(V&&, F &&) -> contraction<V, F>;
} // namespace detail

namespace traits
{
template <typename = void>
struct is_contraction : std::false_type {
};

template <typename V, typename F>
struct is_contraction<detail::contraction<V, F>> : std::true_type {
};

template <typename T>
constexpr bool is_contraction_v = is_contraction<std::remove_cvref_t<T>>::value;
} // namespace traits

template <typename T>
concept Contraction = traits::is_contraction_v<T>;

template <detail::Vector_of_Scalars V, detail::Vector_Aggregrate<V> F>
constexpr auto contract(V&& v, F&& f)
{
    return detail::contraction{std::forward<V>(v), std::forward<F>(f)};
}

} // namespace ccs