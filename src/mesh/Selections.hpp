#pragma once

#include "Cartesian.hpp"
#include "CutGeometry.hpp"
#include "fields/Selector.hpp"
#include "mesh_types.hpp"

#include <range/v3/view/chunk.hpp>
#include <range/v3/view/drop_exactly.hpp>
#include <range/v3/view/for_each.hpp>
#include <range/v3/view/join.hpp>
#include <range/v3/view/stride.hpp>
#include <range/v3/view/take_exactly.hpp>

namespace ccs::mesh::views
{

namespace detail
{
// y-plane view do not result in output iterator when formulated in an intuitve fashion
// using range-v3 building blocks: grid | vs::chunk(nz) | vs::stride(ny) | vs::join; To
// workaround this limitiation, we define a custom y-plane view based on the v3
// view_adapator.  This will need to be revisited if the layout changes
template <typename Rng>
class YPlaneView : public rs::view_adaptor<YPlaneView<Rng>, Rng>
{
    using diff_t = rs::range_difference_t<Rng>;

    friend rs::range_access;
    diff_t nx, ny, nz, j;

    class adaptor : public rs::adaptor_base
    {
        diff_t nx, ny, nz, i, j, k;

    public:
        adaptor() = default;
        adaptor(diff_t nx, diff_t ny, diff_t nz, diff_t i, diff_t j, diff_t k)
            : nx{nx}, ny{ny}, nz{nz}, i{i}, j{j}, k{k}
        {
        }

        template <typename R>
        constexpr auto begin(R& rng)
        {
            auto it = rs::begin(rng.base());
            rs::advance(it, j * nz);
            return it;
        }

        template <typename R>
        constexpr auto end(R& rng)
        {
            auto it = rs::begin(rng.base());
            rs::advance(it, i * ny * nz + j * nz + k);
            return it;
        }

        template <typename I>
        void next(I& it)
        {
            ++k;
            ++it;
            if (k == nz && i != nx - 1) {
                k = 0;
                ++i;
                it += (ny - 1) * nz;
            }
        }

        template <typename I>
        void prev(I& it)
        {
            --k;
            --it;
            if (k < 0) {
                k = nz - 1;
                --i;
                it -= (ny - 1) * nz;
            }
        }

        template <typename I>
        void advance(I& it, rs::difference_type_t<I> n)
        {
            if (n == 0) return;

            const auto line_offset = ny * nz;
            // define a new i and k for the adaptor and adjust iterator accordingly.

            if (n > 0) {
                n += k;

                auto qr = std::div(n, nz);
                diff_t i1 = i + qr.quot;
                diff_t k1 = qr.rem;

                if (i1 == nx) {
                    i1 = nx - 1;
                    k1 = nz;
                }

                rs::advance(it, line_offset * (i1 - i) + (k1 - k));
                i = i1;
                k = k1;
            } else {
                n -= (nz - 1 - k);

                auto qr = std::div(n, nz);
                diff_t i1 = i + qr.quot;
                diff_t k1 = nz - 1 + qr.rem;

                rs::advance(it, line_offset * (i1 - i) + (k1 - k));
                i = i1;
                k = k1;
            }
        }

        template <typename I>
        diff_t distance_to(const I&, const I&, const adaptor& that) const
        {
            return (that.i - i) * nz + (that.k - k);
        }
    };

    adaptor begin_adaptor() { return {nx, ny, nz, 0, j, 0}; }

    adaptor end_adaptor() { return {nx, ny, nz, nx - 1, j, nz}; }

public:
    YPlaneView() = default;
    explicit constexpr YPlaneView(Rng&& rng, const int3& extents, int j)
        : YPlaneView::view_adaptor{FWD(rng)},
          nx{extents[0]},
          ny{extents[1]},
          nz{extents[2]},
          j{j}
    {
    }
};

template <typename Rng>
YPlaneView(Rng&&, int3, int) -> YPlaneView<Rng>;

struct y_plane_base_fn {
    template <typename Rng>
    constexpr auto operator()(Rng&& rng, const int3& extents, int j) const
    {
        return YPlaneView(FWD(rng), extents, j);
    }
};

struct y_plane_fn : y_plane_base_fn {
    using y_plane_base_fn::operator();

    constexpr auto operator()(const int3& extents, int j) const
    {
        return rs::make_view_closure(rs::bind_back(y_plane_base_fn{}, extents, j));
    }
};

constexpr auto y_plane_view = y_plane_fn{};

} // namespace detail

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
struct plane_fn<1> {
    constexpr auto operator()(const int3& extents, int j = 0) const
    {
        return detail::y_plane_view(extents, j);
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
        return field::Tuple{FWD(s) | plane_view<1>(extents)};
    });
}

constexpr auto ymax(int3 extents)
{
    return rs::make_view_closure([extents = MOVE(extents)]<DomainSelection S>(S&& s) {
        return field::Tuple{FWD(s) | plane_view<1>(extents, extents[1] - 1)};
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

constexpr auto location(const Cartesian& cartesian, const CutGeometry& geometry)
{
    using namespace ccs::field::tuple;
    return rs::make_view_closure([&]<traits::SelectionType S>(S&&) {
        if constexpr (traits::is_domain_selection_v<S>)
            return vs::cartesian_product(cartesian.x(), cartesian.y(), cartesian.z());
        else if constexpr (traits::is_Rx_selection_v<S>)
            return geometry.Rx() | vs::transform([](auto&& o) { return o.position; });
        else if constexpr (traits::is_Ry_selection_v<S>)
            return geometry.Ry() | vs::transform([](auto&& o) { return o.position; });
        else if constexpr (traits::is_Rz_selection_v<S>)
            return geometry.Rz() | vs::transform([](auto&& o) { return o.position; });

        else
            static_assert("unaccounted selection type");
    });
}

namespace detail
{
// fluid domain view, need to write our own adaptor so m.F() can be an output view
template <typename Rng>
class FView : public rs::view_adaptor<FView<Rng>, Rng>
{
    using diff_t = rs::range_difference_t<Rng>;

    friend rs::range_access;

    IndexExtents extents;
    std::span<const Line> lines;

    class adaptor : public rs::adaptor_base
    {
        IndexExtents extents;
        std::span<const Line> lines;
        unsigned long line;
        integer i, i0, i1, local_off;

        constexpr void set_line()
        {
            auto&& [_, start, end] = lines[line];

            i0 = start.object_boundary ? extents(start.mesh_coordinate) + 1
                                       : extents(start.mesh_coordinate);
            i1 = end.object_boundary ? extents(end.mesh_coordinate)
                                     : extents(end.mesh_coordinate) + 1;
        }

    public:
        adaptor() = default;
        adaptor(IndexExtents extents, std::span<const Line> lines)
            : extents{MOVE(extents)}, lines{MOVE(lines)}
        {
            assert(this->lines.size() > 0);
        }

        template <typename R>
        constexpr auto begin(R& rng)
        {
            auto it = rs::begin(rng.base());

            local_off = 0;
            line = 0;
            set_line();
            // this doesn't really handle the case of stride > 1
            i = i0;
            rs::advance(it, i);
            return it;
        }

        template <typename R>
        constexpr auto end(R& rng)
        {
            auto it = rs::begin(rng.base());

            local_off = 0;
            for (line = 0; line < lines.size(); line++) {
                set_line();
                local_off += (i1 - i0);
            }
            i = i1;

            rs::advance(it, i);
            return it;
        }

        template <typename I>
        void next(I& it)
        {
            // advance to next point on this line, or to the next line
            ++i;
            ++it;
            ++local_off;
            if (i == i1 && line != lines.size() - 1) {
                ++line;
                set_line();

                it += (i0 - i);
                i = i0;
            }
        }

        template <typename I>
        void prev(I& it)
        {
            --i;
            --it;
            --local_off;
            if (i < i0) {
                --line;
                set_line();

                it -= (i - (i1 - 1));
                i = i1 - 1;
            }
        }

        template <typename I>
        void advance(I& it, rs::difference_type_t<I> n)
        {
            if (n == 0) return;

            local_off += n;
            rs::difference_type_t<I> it_off = 0;
            const auto last_line = lines.size() - 1;

            if (n > 0) {
                // move iterator to i0 to make life easier
                it_off = (i0 - i);
                i = i0;
                n -= it_off;
                while (line != last_line && n > ((i1 - 1) - i0)) {
                    // advance the line and reset i0/i1
                    n -= (i1 - i0);
                    ++line;
                    set_line();
                }
                it_off += i0 + n - i;
                i = i0 + n;
            } else {
                // move iterator to i1 to make life easier
                it_off = (i1 - i);
                i = i1;
                n -= it_off;

                while (n < (i0 - i1)) {
                    n -= (i0 - i1);
                    --line;
                    set_line();
                }
                it_off += i1 + n - i;
                i = i1 + n;
            }

            rs::advance(it, it_off);
        }

        template <typename I>
        diff_t distance_to(const I&, const I&, const adaptor& that) const
        {
            return that.local_off - local_off;
        }
    };

    adaptor begin_adaptor() { return {extents, lines}; }

    adaptor end_adaptor() { return {extents, lines}; }

public:
    FView() = default;
    // This really hints that we should extract the extents into their own object
    // and give them a call operator that converts an ijk tuple to a single index
    // It should also convert so a simple int3
    explicit constexpr FView(Rng&& rng, IndexExtents extents, std::span<const Line> lines)
        : FView::view_adaptor{FWD(rng)}, extents{MOVE(extents)}, lines{MOVE(lines)}
    {
    }
};

template <typename Rng>
FView(Rng&&, IndexExtents, std::span<const Line>) -> FView<Rng>;

struct fview_base_fn {
    template <typename Rng>
    constexpr auto
    operator()(Rng&& rng, IndexExtents extents, std::span<const Line> lines) const
    {
        return FView(FWD(rng), MOVE(extents), MOVE(lines));
    }
};

struct fview_fn : fview_base_fn {
    using fview_base_fn::operator();

    constexpr auto operator()(IndexExtents extents, std::span<const Line> lines) const
    {
        return rs::make_view_closure(
            rs::bind_back(fview_base_fn{}, MOVE(extents), MOVE(lines)));
    }
};

constexpr auto fview = fview_fn{};
} // namespace detail

constexpr auto F(IndexExtents extents, std::span<const Line> lines)
{
    return rs::make_view_closure([=]<DomainSelection S>(S&& s) {
        return field::Tuple{FWD(s) | detail::fview(extents, lines)};
    });
}

} // namespace ccs::mesh::views