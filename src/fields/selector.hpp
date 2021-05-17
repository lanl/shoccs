#pragma once

#include "tuple.hpp"

#include "scalar.hpp"
#include "vector.hpp"

#include "selector_fwd.hpp"

#include "index_extents.hpp"

#include <range/v3/view/drop_exactly.hpp>
#include <range/v3/view/stride.hpp>
#include <range/v3/view/take_exactly.hpp>

namespace ccs
{

// traits for mapping scalar and vector types to appropriate selection mp_list
namespace detail
{
template <typename>
struct selection_fn_index_impl;

template <Scalar T>
struct selection_fn_index_impl<T> {
    using type = mp_size_t<0>;
};

template <Vector T>
struct selection_fn_index_impl<T> {
    using type = mp_size_t<1>;
};
} // namespace detail

template <typename T>
using selection_fn_index = detail::selection_fn_index_impl<T>::type;

template <ListIndex I, Tuple R>
struct selection : R {
    using index = I;

    selection() = default; // default construction needed for semi-regular concept

    constexpr selection(R r) : R{MOVE(r)} {}

    selection(const selection&) = default;
    selection(selection&&) = default;

    selection& operator=(const selection&) = default;
    selection& operator=(selection&&) = default;

    template <typename T>
        requires(!ViewClosures<T>)
    &&std::is_assignable_v<R&, T> selection& operator=(T&& t)
    {
        R::operator=(FWD(t));
        return *this;
    }
    // template <Selection S>
    // selection& operator=(S&& s)
    // {
    //     R::operator=(FWD(s).as_Tuple());
    //     return *this;
    // }

    template <ViewClosures F>
    selection& operator=(F f)
    {
        auto rng = *this | f;
        R::operator=(rng);
        return *this;
    }
};

namespace detail
{
template <ListIndex L, typename R>
constexpr auto make_selection(R r)
{
    return selection<L, R>{MOVE(r)};
}

//
// selection views are designed for extracting the major components of scalars and vectors
// such as the domain - D, or the sets for boundary data - R
//
template <typename... Lists>
struct selection_view_fn {
    using Indices = mp_list<Lists...>;

    template <TupleLike U>
    constexpr auto operator()(U&& u) const
    {

        // for now just grab the first element of the list
        using List = mp_at<Indices, selection_fn_index<U>>;
        static_assert(!mp_empty<List>::value, "selection operation not permitted");

        // Build a selection using the tuples indexed in the list
        return []<auto... Is>(std::index_sequence<Is...>, auto&& u)
        {
            if constexpr (sizeof...(Is) == 1)
                return detail::make_selection<mp_at_c<List, 0>>(
                    tuple{get<mp_at_c<List, 0>>(FWD(u))});
            else
                return tuple{detail::make_selection<mp_at_c<List, Is>>(
                    tuple{get<mp_at_c<List, Is>>(FWD(u))})...};
        }
        (sequence<List>, FWD(u));
    }
};

} // namespace detail

template <typename... Lists>
inline constexpr auto
    selection_view = rs::make_view_closure(detail::selection_view_fn<Lists...>{});

//
// selectors for main components of scalars and vectors
//

namespace sel
{
inline constexpr auto Dx = selection_view<mp_list<>, mp_list<vi::Dx>>;
inline constexpr auto Dy = selection_view<mp_list<>, mp_list<vi::Dy>>;
inline constexpr auto Dz = selection_view<mp_list<>, mp_list<vi::Dz>>;
inline constexpr auto Rx =
    selection_view<mp_list<si::Rx>, mp_list<vi::xRx, vi::yRx, vi::zRx>>;
inline constexpr auto Ry =
    selection_view<mp_list<si::Ry>, mp_list<vi::xRy, vi::yRy, vi::zRy>>;
inline constexpr auto Rz =
    selection_view<mp_list<si::Rz>, mp_list<vi::xRz, vi::yRz, vi::zRz>>;
inline constexpr auto D = selection_view<mp_list<si::D>, mp_list<vi::Dx, vi::Dy, vi::Dz>>;
inline constexpr auto R = selection_view<mp_list<si::Rx, si::Ry, si::Rz>,
                                         mp_list<vi::xRx,
                                                 vi::yRx,
                                                 vi::zRx,
                                                 vi::xRy,
                                                 vi::yRy,
                                                 vi::zRy,
                                                 vi::xRz,
                                                 vi::yRz,
                                                 vi::zRz>>;
inline constexpr auto xRx = selection_view<mp_list<>, mp_list<vi::xRx>>;
inline constexpr auto xRy = selection_view<mp_list<>, mp_list<vi::xRy>>;
inline constexpr auto xRz = selection_view<mp_list<>, mp_list<vi::xRz>>;
inline constexpr auto yRx = selection_view<mp_list<>, mp_list<vi::yRx>>;
inline constexpr auto yRy = selection_view<mp_list<>, mp_list<vi::yRy>>;
inline constexpr auto yRz = selection_view<mp_list<>, mp_list<vi::yRz>>;
inline constexpr auto zRx = selection_view<mp_list<>, mp_list<vi::zRx>>;
inline constexpr auto zRy = selection_view<mp_list<>, mp_list<vi::zRy>>;
inline constexpr auto zRz = selection_view<mp_list<>, mp_list<vi::zRz>>;

} // namespace sel

namespace detail
{

// y-plane view do not result in output iterator when formulated in an intuitve fashion
// using range-v3 building blocks: grid | vs::chunk(nz) | vs::stride(ny) | vs::join; To
// workaround this limitiation, we define a custom y-plane view based on the v3
// view_adapator.  This will need to be revisited if the layout changes
template <typename Rng>
class y_plane_view : public rs::view_adaptor<y_plane_view<Rng>, Rng>
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
    y_plane_view() = default;
    explicit constexpr y_plane_view(Rng&& rng, index_extents extents, int j)
        : y_plane_view::view_adaptor{FWD(rng)},
          nx{extents[0]},
          ny{extents[1]},
          nz{extents[2]},
          j{j}
    {
    }
};

template <typename Rng>
y_plane_view(Rng&&, index_extents, int) -> y_plane_view<Rng>;

template <auto I>
struct plane_selection_base_fn;

template <>
struct plane_selection_base_fn<0> {
    template <Range R>
    constexpr auto operator()(R&& r, index_extents extents, int i) const
    {
        auto n = extents[1] * extents[2];
        return FWD(r) | vs::drop_exactly(i * n) | vs::take_exactly(n);
    }
};

template <>
struct plane_selection_base_fn<1> {
    template <Range R>
    constexpr auto operator()(R&& r, index_extents extents, int j) const
    {
        return y_plane_view(FWD(r), MOVE(extents), j);
    }
};

template <>
struct plane_selection_base_fn<2> {
    template <Range R>
    constexpr auto operator()(R&& r, index_extents extents, int k) const
    {
        auto n = extents[2];
        return FWD(r) | vs::drop_exactly(k) | vs::stride(n);
    }
};

// First parameter indicate the direction of the plane {0, 1, 2}
template <auto I>
struct plane_selection_fn : plane_selection_base_fn<I> {
    using base = plane_selection_base_fn<I>;

    template <TupleLike U>
    constexpr auto operator()(U&& u, index_extents extents, int plane_coord) const
    {
        if (plane_coord < 0) { plane_coord += extents[I]; }
        if constexpr (Scalar<U>) {
            return tuple{base::operator()(FWD(u) | sel::D, MOVE(extents), plane_coord)};
        } else {
            return tuple{base::operator()(FWD(u), MOVE(extents), plane_coord)};
        }
    }

    constexpr auto operator()(index_extents extents, int plane_coord) const
    {
        return rs::make_view_closure(rs::bind_back(*this, MOVE(extents), plane_coord));
    }

    constexpr auto operator()(int plane_coord) const
    {
        return rs::bind_back(*this, plane_coord);
    }
};
} // namespace detail

template <auto I>
constexpr auto plane_selection_fn = detail::plane_selection_fn<I>{};

//
// Selectors for planes of data for Tuples, Scalars, and Vectors
//

namespace sel
{

inline constexpr auto x_plane = plane_selection_fn<0>;
inline constexpr auto y_plane = plane_selection_fn<1>;
inline constexpr auto z_plane = plane_selection_fn<2>;

inline constexpr auto xmin = x_plane(0);
inline constexpr auto xmax = x_plane(-1);
inline constexpr auto ymin = y_plane(0);
inline constexpr auto ymax = y_plane(-1);
inline constexpr auto zmin = z_plane(0);
inline constexpr auto zmax = z_plane(-1);

using xmin_t = decltype(sel::xmin(index_extents{}));
using xmax_t = decltype(sel::xmax(index_extents{}));
using ymin_t = decltype(sel::ymin(index_extents{}));
using ymax_t = decltype(sel::ymax(index_extents{}));
using zmin_t = decltype(sel::zmin(index_extents{}));
using zmax_t = decltype(sel::zmax(index_extents{}));

} // namespace sel

namespace detail
{
// multi_slice view is used to select multiple slices of data and treat them as a single
// range. Used to construct the `fluid` selector
template <typename Rng>
class multi_slice_view : public rs::view_adaptor<multi_slice_view<Rng>, Rng>
{
    using diff_t = rs::range_difference_t<Rng>;

    friend rs::range_access;

    std::span<const index_slice> slices;

    class adaptor : public rs::adaptor_base
    {
        using slice_it = typename std::span<const index_slice>::iterator;

        // iterators to keep track of the slices
        slice_it slice;
        slice_it last_slice;

        integer i;       // current index in the base range [slice->first, slice->last)
        integer multi_i; // index in the multi_slice, allows for quick size computation

        // constexpr void set_line()
        // {
        //     auto&& [_, start, end] = lines[l];

        //     i0 = start.object ? extents(start.mesh_coordinate) + 1
        //                       : extents(start.mesh_coordinate);
        //     i1 = end.object ? extents(end.mesh_coordinate)
        //                     : extents(end.mesh_coordinate) + 1;
        // }

    public:
        adaptor() = default;
        adaptor(std::span<const index_slice> slices)
            : slice{rs::begin(slices)}, last_slice{rs::end(slices)}
        {
        }

        template <typename R>
        constexpr auto begin(R& rng)
        {
            auto it = rs::begin(rng.base());

            multi_i = 0;
            i = slice != last_slice ? slice->first : 0;

            rs::advance(it, i);
            return it;
        }

        template <typename R>
        constexpr auto end(R& rng)
        {
            auto it = rs::begin(rng.base());

            for (multi_i = 0; slice != last_slice; ++slice) {
                multi_i += slice->last - slice->first;
                i = slice->last;
            }

            rs::advance(it, i);
            return it;
        }

        template <typename I>
        void next(I& it)
        {
            // advance to next point on this line, or to the next line
            ++i;
            ++it;
            ++multi_i;
            if (i == slice->last && ++slice != last_slice) {
                it += (slice->first - i);
                i = slice->first;
            }
        }

        template <typename I>
        void prev(I& it)
        {
            --i;
            --it;
            --multi_i;
            if (i < slice->first) {
                --slice;
                it -= (i - (slice->last - 1));
                i = slice->last - 1;
            }
        }

        template <typename I>
        void advance(I& it, rs::difference_type_t<I> n)
        {
            if (n == 0) return;

            multi_i += n;
            rs::difference_type_t<I> it_off = 0;
            // const auto last_line = lines.size() - 1;

            if (n > 0) {
                // move iterator to i0 to make life easier
                it_off = (slice->first - i);
                i = slice->first;
                n -= it_off;
                while (slice != last_slice && n > ((slice->last - 1) - slice->first)) {
                    // advance the line and reset i0/i1
                    n -= (slice->last - slice->first);
                    ++slice;
                    // set_line();
                }
                it_off += slice->first + n - i;
                i = slice->first + n;
            } else {
                // move iterator to i1 to make life easier
                it_off = (slice->last - i);
                i = slice->last;
                n -= it_off;

                while (n < (slice->first - slice->last)) {
                    n -= (slice->first - slice->last);
                    --slice;
                    // set_line();
                }
                it_off += slice->last + n - i;
                i = slice->last + n;
            }

            rs::advance(it, it_off);
        }

        template <typename I>
        diff_t distance_to(const I&, const I&, const adaptor& that) const
        {
            return that.multi_i - multi_i;
        }
    };

    adaptor begin_adaptor() { return {slices}; }

    adaptor end_adaptor() { return {slices}; }

public:
    multi_slice_view() = default;

    explicit constexpr multi_slice_view(Rng&& rng, std::span<const index_slice> slices)
        : multi_slice_view::view_adaptor{FWD(rng)}, slices{MOVE(slices)}
    {
    }
};

template <typename Rng>
multi_slice_view(Rng&&, std::span<const index_slice>) -> multi_slice_view<Rng>;

struct multi_slice_base_fn {
    template <typename Rng>
    constexpr auto operator()(Rng&& rng, std::span<const index_slice> slices) const
    {
        return multi_slice_view(FWD(rng), MOVE(slices));
    }
};

struct multi_slice_fn : multi_slice_base_fn {
    using base = multi_slice_base_fn;

    template <TupleLike U>
    constexpr auto operator()(U&& u, std::span<const index_slice> slices) const
    {
        if constexpr (Scalar<U>) {
            return tuple{base::operator()(FWD(u) | sel::D, MOVE(slices))};
        } else {
            return tuple{base::operator()(FWD(u), MOVE(slices))};
        }
    }

    constexpr auto operator()(std::span<const index_slice> slices) const
    {
        return rs::make_view_closure(rs::bind_back(*this, MOVE(slices)));
    }
};
} // namespace detail

namespace sel
{
constexpr inline auto multi_slice = ::ccs::detail::multi_slice_fn{};
using multi_slice_t = decltype(multi_slice(std::span<const index_slice>{}));
} // namespace sel

} // namespace ccs
