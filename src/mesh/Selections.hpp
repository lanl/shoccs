#pragma once

#include "fields/Selector.hpp"

#include <range/v3/view/chunk.hpp>
#include <range/v3/view/drop_exactly.hpp>
#include <range/v3/view/for_each.hpp>
#include <range/v3/view/join.hpp>
#include <range/v3/view/stride.hpp>
#include <range/v3/view/take_exactly.hpp>

namespace ccs::mesh::views
{

template <auto I>
struct plane_fn;

template <>
struct plane_fn<0> {
    constexpr auto operator()(const int3& extents) const
    {
        return vs::take_exactly(extents[1] * extents[2]);
    }

    constexpr auto operator()(const int3& extents, int i) const
    {
        return vs::drop_exactly(i * extents[1] * extents[2]) | (*this)(extents);
    }
};

template <>
struct plane_fn<2> {
    constexpr auto operator()(const int3& extents) const
    {
        return vs::stride(extents[2]);
    }

    constexpr auto operator()(const int3& extents, int k) const
    {
        return vs::drop_exactly(k) | (*this)(extents);
    }
};

template <auto I>
constexpr auto plane_view = plane_fn<I>{};

using field::tuple::traits::DomainSelection;

constexpr auto xmin(int3 extents)
{
    return rs::make_view_closure([extents = MOVE(extents)]<DomainSelection S>(S&& s) {
        return field::Tuple{FWD(s) | plane_view<0>(extents)};
    });
}

constexpr auto xmax(int3 extents)
{
    return rs::make_view_closure([extents = MOVE(extents)]<DomainSelection S>(S&& s) {
        return field::Tuple{FWD(s) | plane_view<0>(extents, extents[0] - 1)};
    });
}

constexpr auto ymin(int3 extents)
{
    return rs::make_view_closure([extents = MOVE(extents)]<DomainSelection S>(S&& s) {
        auto [nx, ny, nz] = extents;
        // return field::Tuple{FWD(s) | vs::chunk(nz) | vs::stride(ny) |
        // vs::join};
        return field::Tuple{FWD(s) | vs::chunk(ny * nz) |
                            vs::for_each(vs::take_exactly(nz))};
    });
}

constexpr auto ymax(int3 extents)
{
    return rs::make_view_closure([extents = MOVE(extents)]<DomainSelection S>(S&& s) {
        auto [nx, ny, nz] = extents;
        return field::Tuple{FWD(s) | vs::drop_exactly((ny - 1) * nz) | vs::chunk(nz) |
                            vs::stride(ny) | vs::join};
    });
}

constexpr auto zmin(int3 extents)
{
    return rs::make_view_closure([extents = MOVE(extents)]<DomainSelection S>(S&& s) {
        return field::Tuple{FWD(s) | plane_view<2>(extents)};
    });
}

constexpr auto zmax(int3 extents)
{
    return rs::make_view_closure([extents = MOVE(extents)]<DomainSelection S>(S&& s) {
        return field::Tuple{FWD(s) | plane_view<2>(extents, extents[2] - 1)};
    });
}

} // namespace ccs::mesh::views