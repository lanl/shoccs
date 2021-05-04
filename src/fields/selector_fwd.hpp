#pragma once

#include "tuple_fwd.hpp"
#include "types.hpp"

namespace ccs::si
{

using D = list_index<0, 0>;
using Rx = list_index<1, 0>;
using Ry = list_index<1, 1>;
using Rz = list_index<1, 2>;
} // namespace ccs::si

namespace ccs::vi
{
// using ccs::field::tuple::traits::list_index;

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
} // namespace ccs::vi

namespace ccs
{
template <ListIndex, Tuple R>
struct selection;

namespace detail
{
template <typename>
struct is_selection_impl : std::false_type {
};

template <ListIndex I, Tuple R>
struct is_selection_impl<selection<I, R>> : std::true_type {
};
} // namespace detail

template <typename S>
using is_selection = detail::is_selection_impl<std::remove_cvref_t<S>>::type;

template <typename S>
concept Selection = is_selection<S>::value;

// // traits for extracting the selection indicies
template <Selection S>
using selection_index = std::remove_cvref_t<S>::index;

template <Selection S>
using is_domain_selection =
    mp_set_contains<mp_list<si::D, vi::Dx, vi::Dy, vi::Dz>, selection_index<S>>;

template <Selection S>
using is_Rx_selection =
    mp_set_contains<mp_list<si::Rx, vi::xRx, vi::yRx, vi::zRx>, selection_index<S>>;

template <Selection S>
using is_Ry_selection =
    mp_set_contains<mp_list<si::Ry, vi::xRy, vi::yRy, vi::zRy>, selection_index<S>>;

template <Selection S>
using is_Rz_selection =
    mp_set_contains<mp_list<si::Rz, vi::xRz, vi::yRz, vi::zRz>, selection_index<S>>;

template <typename T>
constexpr auto is_domain_selection_v = is_domain_selection<T>::value;
template <typename T>
constexpr auto is_Rx_selection_v = is_Rx_selection<T>::value;
template <typename T>
constexpr auto is_Ry_selection_v = is_Ry_selection<T>::value;
template <typename T>
constexpr auto is_Rz_selection_v = is_Rz_selection<T>::value;

template <typename T>
concept DomainSelection = Selection<T> && is_domain_selection_v<T>;

} // namespace ccs

namespace ranges
{
template <ccs::ListIndex L, ccs::Tuple R>
inline constexpr bool enable_view<ccs::selection<L, R>> = enable_view<R>;
}
