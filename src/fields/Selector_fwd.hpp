#pragma once

#include "Tuple_fwd.hpp"
#include "types.hpp"

namespace ccs::selector::scalar
{
using ccs::field::tuple::traits::list_index;

using D = list_index<0, 0>;
using Rx = list_index<1, 0>;
using Ry = list_index<1, 1>;
using Rz = list_index<1, 2>;
} // namespace ccs::selector::scalar

namespace ccs::selector::vector
{
using ccs::field::tuple::traits::list_index;

using X = list_index<0>;
using Y = list_index<1>;
using Z = list_index<2>;
using Dx = list_index<0, 0, 0>;
using Dy = list_index<1, 0, 0>;
using Dz = list_index<2, 0, 0>;
using xRx = list_index<0, 1, 0>;
using xRy = list_index<0, 1, 1>;
using xRz = list_index<0, 1, 2>;
using yRx = list_index<1, 1, 0>;
using yRy = list_index<1, 1, 1>;
using yRz = list_index<1, 1, 2>;
using zRx = list_index<2, 1, 0>;
using zRy = list_index<2, 1, 1>;
using zRz = list_index<2, 1, 2>;
} // namespace ccs::selector::vector

namespace ccs::field::tuple
{
template <traits::ListIndex, traits::TupleType R>
struct Selection;

namespace traits
{
namespace detail
{
template <typename>
struct is_selection_impl : std::false_type {
};

template <traits::ListIndex I, traits::TupleType R>
struct is_selection_impl<Selection<I, R>> : std::true_type {
};
} // namespace detail

template <typename S>
using is_Selection = detail::is_selection_impl<std::remove_cvref_t<S>>::type;

template <typename S>
concept SelectionType = is_Selection<S>::value;

// // traits for extracting the selection indicies
template <SelectionType S>
using selection_index = std::remove_cvref_t<S>::Index;

template <SelectionType S>
using is_domain_selection = mp_set_contains<mp_list<selector::scalar::D,
                                                    selector::vector::Dx,
                                                    selector::vector::Dy,
                                                    selector::vector::Dz>,
                                            selection_index<S>>;
template <SelectionType S>
using is_Rx_selection = mp_set_contains<mp_list<selector::scalar::Rx,
                                                selector::vector::xRx,
                                                selector::vector::yRx,
                                                selector::vector::zRx>,
                                        selection_index<S>>;
template <SelectionType S>
using is_Ry_selection = mp_set_contains<mp_list<selector::scalar::Ry,
                                                selector::vector::xRy,
                                                selector::vector::yRy,
                                                selector::vector::zRy>,
                                        selection_index<S>>;

template <SelectionType S>
using is_Rz_selection = mp_set_contains<mp_list<selector::scalar::Rz,
                                                selector::vector::xRz,
                                                selector::vector::yRz,
                                                selector::vector::zRz>,
                                        selection_index<S>>;

template <typename T>
constexpr auto is_domain_selection_v = is_domain_selection<T>::value;
template <typename T>
constexpr auto is_Rx_selection_v = is_Rx_selection<T>::value;
template <typename T>
constexpr auto is_Ry_selection_v = is_Ry_selection<T>::value;
template <typename T>
constexpr auto is_Rz_selection_v = is_Rz_selection<T>::value;

template <typename T>
concept DomainSelection = SelectionType<T>&& is_domain_selection_v<T>;

} // namespace traits
} // namespace ccs::field::tuple

namespace ranges
{
template <ccs::field::tuple::traits::ListIndex L, ccs::field::tuple::traits::TupleType R>
inline constexpr bool enable_view<ccs::field::tuple::Selection<L, R>> = enable_view<R>;
}