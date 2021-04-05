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

} // namespace ccs::mesh::views