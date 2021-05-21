#pragma once

#include "selector.hpp"

namespace ccs
{

namespace detail
{
template <typename T>
class field;
}

template <typename>
struct is_field : std::false_type {
};

template <typename T>
struct is_field<detail::field<T>> : std::true_type {
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

} // namespace ccs
