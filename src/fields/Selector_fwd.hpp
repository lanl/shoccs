#pragma once

#include "Tuple_fwd.hpp"
#include "types.hpp"

namespace ccs::field::tuple
{
template <traits::TupleType R, auto...>
struct Selection;

namespace traits
{
namespace detail
{
template <typename>
struct is_selection_impl : std::false_type {
};

template <traits::TupleType R, auto... I>
struct is_selection_impl<Selection<R, I...>> : std::true_type {
};
} // namespace detail

template <typename S>
using is_Selection = detail::is_selection_impl<std::remove_cvref_t<S>>::type;

template <typename S>
concept SelectionType = is_Selection<S>::value;

// traits for extracting the selection indicies
template <SelectionType S>
using selection_indices = std::remove_cvref_t<S>::Idx;

template <SelectionType S>
using selection_index_length = mp_size<selection_indices<S>>;

template <SelectionType S>
constexpr auto selection_index_length_v = selection_index_length<S>::value;

template <SelectionType S>
using last_selection_index = mp_back<selection_indices<S>>;

template <SelectionType S>
constexpr auto last_selection_index_v = last_selection_index<S>::value;

template <SelectionType S, std::size_t I>
using selection_index = mp_at_c<selection_indices<S>, I>;

template <SelectionType S, std::size_t I>
constexpr auto selection_index_v = selection_index<S, I>::value;

} // namespace traits
} // namespace ccs::field::tuple
