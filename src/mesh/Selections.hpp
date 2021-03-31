#pragma once

#include "fields/Selector.hpp"

#include <range/v3/view/chunk.hpp>
#include <range/v3/view/drop_exactly.hpp>
#include <range/v3/view/for_each.hpp>
#include <range/v3/view/join.hpp>
#include <range/v3/view/stride.hpp>
#include <range/v3/view/take_exactly.hpp>

namespace ccs::mesh::detail
{
constexpr auto xmin(int3 extents)
{
    return rs::make_view_closure(
        [extents = MOVE(extents)]<field::tuple::traits::SelectionType S>(S&& s) {
            static_assert(field::tuple::traits::is_domain_selection_v<S>);
            auto [nx, ny, nz] = extents;
            return field::Tuple{FWD(s) | vs::take_exactly(ny * nz)};
        });
}

constexpr auto xmax(int3 extents)
{
    return rs::make_view_closure(
        [extents = MOVE(extents)]<field::tuple::traits::SelectionType S>(S&& s) {
            static_assert(field::tuple::traits::is_domain_selection_v<S>);
            auto [nx, ny, nz] = extents;
            return field::Tuple{FWD(s) | vs::drop_exactly((nx - 1) * ny * nz)};
        });
}

constexpr auto ymin(int3 extents)
{
    return rs::make_view_closure(
        [extents = MOVE(extents)]<field::tuple::traits::SelectionType S>(S&& s) {
            static_assert(field::tuple::traits::is_domain_selection_v<S>);
            auto [nx, ny, nz] = extents;
            // return field::Tuple{FWD(s) | vs::chunk(nz) | vs::stride(ny) | vs::join};
            return field::Tuple{FWD(s) | vs::chunk(ny * nz) |
                                vs::for_each(vs::take_exactly(nz))};
        });
}

constexpr auto ymax(int3 extents)
{
    return rs::make_view_closure(
        [extents = MOVE(extents)]<field::tuple::traits::SelectionType S>(S&& s) {
            static_assert(field::tuple::traits::is_domain_selection_v<S>);
            auto [nx, ny, nz] = extents;
            return field::Tuple{FWD(s) | vs::drop_exactly((ny - 1) * nz) | vs::chunk(nz) |
                                vs::stride(ny) | vs::join};
        });
}

constexpr auto zmin(int3 extents)
{
    return rs::make_view_closure(
        [extents = MOVE(extents)]<field::tuple::traits::SelectionType S>(S&& s) {
            static_assert(field::tuple::traits::is_domain_selection_v<S>);
            auto [nx, ny, nz] = extents;
            return field::Tuple{FWD(s) | vs::stride(nz)};
        });
}

constexpr auto zmax(int3 extents)
{
    return rs::make_view_closure(
        [extents = MOVE(extents)]<field::tuple::traits::SelectionType S>(S&& s) {
            static_assert(field::tuple::traits::is_domain_selection_v<S>);
            auto [nx, ny, nz] = extents;
            return field::Tuple{FWD(s) | vs::drop_exactly(nz - 1) | vs::stride(nz)};
        });
}

} // namespace ccs::mesh::detail