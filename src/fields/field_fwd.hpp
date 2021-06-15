#pragma once

#include "selector.hpp"

namespace ccs
{

struct system_size {
    integer nscalars;
    integer nvectors;
    scalar<integer> scalar_size;

    constexpr friend auto operator<=>(const system_size&, const system_size&) = default;
};

namespace detail
{
template <Range S, Range V>
class field;
}

template <typename>
struct is_field : std::false_type {
};

template <Range S, Range V>
struct is_field<detail::field<S, V>> : std::true_type {
};

template <typename T>
concept Field = is_field<std::remove_cvref_t<T>>::value;

namespace detail
{
    template <typename T>
    struct field_scalar_impl {
        using type = decltype(std::declval<T>().scalars()[0]);
    };

} // namespace detail

template <Field T>
using field_scalar = detail::field_scalar_impl<T>::type;

template <typename Out, typename In>
concept OutputField = Field<Out> &&
    ((!Field<In> && OutputTuple<field_scalar<Out>, In>) ||
     (Field<In> && OutputTuple<field_scalar<Out>, field_scalar<In>>));

namespace detail
{
template <typename>
struct scalar_type_impl;

template <typename S, typename V>
struct scalar_type_impl<field<S, V>> {
    using type = S;
};

template <typename>
struct vector_type_impl;

template <typename S, typename V>
struct vector_type_impl<field<S, V>> {
    using type = V;
};
} // namespace detail

template <Field F>
using scalar_type = detail::scalar_type_impl<std::remove_cvref_t<F>>::type;

template <Field F>
using vector_type = detail::vector_type_impl<std::remove_cvref_t<F>>::type;

template <Field F>
using scalar_value_t = rs::range_value_t<scalar_type<F>>;

template <Field F>
using vector_value_t = rs::range_value_t<vector_type<F>>;

template <Field F>
using scalar_ref_t = rs::range_reference_t<scalar_type<F>>;

template <Field F>
using vector_ref_t = rs::range_reference_t<vector_type<F>>;
} // namespace ccs
