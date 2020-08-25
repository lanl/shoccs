#include "geometry.hpp"
#include "indexing.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

namespace ccs
{

static int npoints(const umesh_line& line)
{
    return 1 + static_cast<int>((line.max - line.min + 0.1 * line.h) / line.h);
}

static std::optional<hit_info>
closest_hit(std::span<const shape> shapes, const ray& r, real t_min, real t_max)
{
    std::optional<hit_info> global_hit{};
    // find minimum t for mesh/object intersection
    for (auto&& s : shapes) {
        if (auto current_hit = s.hit(r, t_min, t_max); current_hit) {
            global_hit = current_hit;
            // adjust t_max and see if we can find something closer
            t_max = current_hit->t;
        }
    }
    return global_hit;
}

// check for intersections along line I using line
template <int I>
static void init_line(std::span<const shape> shapes,
                      const umesh_line& xline,
                      const umesh_line& yline,
                      const umesh_line& zline,
                      std::vector<mesh_object_info>& info,
                      std::vector<std::vector<mesh_object_info>>& sorted_info)
{
    sorted_info.resize(shapes.size());
    // handy shortcuts
    using T = const umesh_line&;
    const std::tuple<T, T, T> lines{xline, yline, zline};
    constexpr auto S = index::dir<I>::slow;
    constexpr auto F = index::dir<I>::fast;

    T fline = std::get<F>(lines);
    T sline = std::get<S>(lines);
    T iline = std::get<I>(lines);

    const int nf = npoints(fline);
    const int ns = npoints(sline);

    real3 origin{};
    int3 coord{};

    for (int s = 0; s < ns; s++) {
        origin[S] = sline.min + s * sline.h;
        coord[S] = s;

        for (int f = 0; f < nf; f++) {
            origin[F] = fline.min + f * fline.h;
            coord[F] = f;

            const auto& [min, max, h] = iline;

            real t_min{0};
            real t_max{max - min};

            origin[I] = min;

            real3 direction{};
            direction[I] = 1.0;

            const ray r{origin, direction};
            while (auto hit = closest_hit(shapes, r, t_min, t_max)) {
                // how should this be handled to favor uniform over degenerate cases.
                coord[I] = static_cast<int>(hit->t / iline.h) + hit->ray_outside;

                // if ray_outside then coord[I]-1 is the fluid coord and psi = hit->position[I]-(mesh_position[coord[I]-1])
                // if !ray_outside then coord[I]+1 is the fluid coord and psi = mesh_position[coord[I]+1] - hit->position[I]
                int off = 1 - 2 * hit->ray_outside;
                real fluid_pos = min + h * (coord[I] + off);
                real psi = off * (fluid_pos - hit->position[I]) / h;

                auto id = hit->shape_id;
                info.push_back(
                    mesh_object_info{psi, hit->position, hit->ray_outside, coord, id});
                sorted_info[id].push_back(
                    mesh_object_info{psi, hit->position, hit->ray_outside, coord, id});

                t_min = std::nextafter(hit->t, t_max);
            }
        }
    }
}

template <int J, int K>
static constexpr bool same_plane(const int3& x, const int3& y)
{
    return x[J] == y[J] && x[K] == y[K];
}

// append a range of points in `I` direction for:
// [starting_coord, ending_I]
template <int I>
static void append_solid_points(std::vector<int3>& info,
                                int3 starting_coord,
                                int ending_I)
{
    int nitems = ending_I - starting_coord[I]+1;
    // info.reserve(info.size() + nitems);
    for (int i = 0; i < nitems; i++) {
        info.push_back(starting_coord);
        ++starting_coord[I];
    }

    assert(nitems < 0 || starting_coord[I] == ending_I+1);
}

template <int I>
static void init_solid(const umesh_line& xline,
                       const umesh_line& yline,
                       const umesh_line& zline,
                       std::span<const mesh_object_info> r,
                       std::vector<int3>& info)
{
    using T = const umesh_line&;
    const std::tuple<T, T, T> lines{xline, yline, zline};
    constexpr auto S = index::dir<I>::slow;
    constexpr auto F = index::dir<I>::fast;

    T iline = std::get<I>(lines);

    // npoints is needed to bound the calculation
    const int ni = npoints(iline);

    // Loop through all solid_coord (SC) in Rw.   These are the boundaries of the solid
    // points to be added to info object.  There are Z cases to consider
    // 1.) We encounter a SC which has `!ray_outside`.
    //     A.) If `last_coord` refers to a point on this line then all points between
    //         `last_coord` and this point are solid
    //     B.) If `last_coord` does not refer to a point on this line, all points
    //         from the start of this line to the current point are solid
    // 2.) We encounter a SC which has `ray_outside`
    //     - how many points to mark as solid depend on the next point.
    //     A.) If the next point is on the same line, the case is handled on the next
    //         loop iteration as 1A.
    //     B.) If the next point is not on the same line, then the points after the
    //         current coordinate should all be marked solid.

    auto first = r.begin();
    auto last = r.end();
    auto prev = last;

    while (first != last) {
        const mesh_object_info& m = *first;

        if (m.ray_outside) {
            auto next = first + 1;
            if (next == last || !same_plane<S, F>(m.solid_coord, (*next).solid_coord)) {
                append_solid_points<I>(info, m.solid_coord, ni);
            }
        } else if (prev == last) {
            // if prev == last, then this is the first intersection point encountered

            // All points upto the current point are solid
            int3 origin{};
            // we don't handle fully solid lines correctly so assert that there
            // aren't any
            assert(m.solid_coord[S] == 0 && m.solid_coord[F] == 0);
            append_solid_points<I>(info, origin, m.solid_coord[I]);

        } else if (same_plane<S, F>((*prev).solid_coord, m.solid_coord)) {
            append_solid_points<I>(info, (*prev).solid_coord, m.solid_coord[I]);
        } else {
            int3 origin{};
            origin[S] = m.solid_coord[S];
            origin[F] = m.solid_coord[F];
            append_solid_points<I>(info, origin, m.solid_coord[I]);
        }

        prev = first++;
    }
}

geometry::geometry(std::span<const shape> shapes,
                   const umesh_line& xline,
                   const umesh_line& yline,
                   const umesh_line& zline)
{
    init_line<0>(shapes, xline, yline, zline, rx_, rx_m_);
    init_line<1>(shapes, xline, yline, zline, ry_, ry_m_);
    init_line<2>(shapes, xline, yline, zline, rz_, rz_m_);

    init_solid<0>(xline, yline, zline, rx_, sx_);
    init_solid<1>(xline, yline, zline, ry_, sy_);
    init_solid<2>(xline, yline, zline, rz_, sz_);
}

std::span<const mesh_object_info> geometry::Rx() const { return rx_; }

std::span<const mesh_object_info> geometry::Rx(int shape_id) const
{
    return rx_m_[shape_id];
}

std::span<const mesh_object_info> geometry::Ry() const { return ry_; }

std::span<const mesh_object_info> geometry::Ry(int shape_id) const
{
    return ry_m_[shape_id];
}

std::span<const mesh_object_info> geometry::Rz() const { return rz_; }

std::span<const mesh_object_info> geometry::Rz(int shape_id) const
{
    return rz_m_[shape_id];
}

std::span<const int3> geometry::Sx() const { return sx_; }
std::span<const int3> geometry::Sy() const { return sy_; }
std::span<const int3> geometry::Sz() const { return sz_; }

} // namespace ccs