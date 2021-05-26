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
template <ListIndex, Tuple, typename>
struct selection;

} // namespace ccs

namespace ranges
{
template <ccs::ListIndex L, ccs::Tuple R, typename Fn>
inline constexpr bool enable_view<ccs::selection<L, R, Fn>> = enable_view<R>;
}
