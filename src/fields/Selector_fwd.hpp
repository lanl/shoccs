#pragma once

#include "Tuple_fwd.hpp"
#include "types.hpp"

namespace ccs::selector
{
template <typename L, field::tuple::traits::TupleType R>
struct Selection;

namespace traits
{
template <typename>
struct is_Selection : std::false_type {
};

template <typename L, field::tuple::traits::TupleType R>
struct is_Selection<Selection<L, R>> : std::true_type {
};

template <typename S>
concept SelectionType = is_Selection<std::remove_cvref_t<S>>::value;
} // namespace traits
} // namespace ccs::selector