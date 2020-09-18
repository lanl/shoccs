#include "geometry.hpp"
#include "indexing.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

namespace ccs
{

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
                      const std::array<umesh_line, 3>& lines,
                      std::vector<mesh_object_info>& info,
                      std::vector<std::vector<mesh_object_info>>& sorted_info)
{
    sorted_info.resize(shapes.size());
    // handy shortcuts
    constexpr auto S = index::dir<I>::slow;
    constexpr auto F = index::dir<I>::fast;

    const umesh_line& fline = lines[F];
    const umesh_line& sline = lines[S];
    const umesh_line& iline = lines[I];

    real3 origin{};
    int3 coord{};

    for (int s = 0; s < sline.n; s++) {
        origin[S] = sline.min + s * sline.h;
        coord[S] = s;

        for (int f = 0; f < fline.n; f++) {
            origin[F] = fline.min + f * fline.h;
            coord[F] = f;

            const auto& [min, max, h, n] = iline;

            real t_min{0};
            real t_max{max - min};

            origin[I] = min;

            real3 direction{};
            direction[I] = 1.0;

            const ray r{origin, direction};
            while (auto hit = closest_hit(shapes, r, t_min, t_max)) {
                // how should this be handled to favor uniform over degenerate cases.
                coord[I] = static_cast<int>(hit->t / iline.h) + hit->ray_outside;

                // if ray_outside then coord[I]-1 is the fluid coord and psi =
                // hit->position[I]-(mesh_position[coord[I]-1]) if !ray_outside then
                // coord[I]+1 is the fluid coord and psi = mesh_position[coord[I]+1] -
                // hit->position[I]
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
static void
append_solid_points(std::vector<int3>& info, int3 starting_coord, int ending_I)
{
    int nitems = ending_I - starting_coord[I] + 1;
    // info.reserve(info.size() + nitems);
    for (int i = 0; i < nitems; i++) {
        info.push_back(starting_coord);
        ++starting_coord[I];
    }

    assert(nitems < 0 || starting_coord[I] == ending_I + 1);
}

template <int I>
static void init_solid(const std::array<umesh_line, 3>& lines,
                       std::span<const mesh_object_info> r,
                       std::vector<int3>& info)
{
    constexpr auto S = index::dir<I>::slow;
    constexpr auto F = index::dir<I>::fast;

    const umesh_line& iline = lines[I];

    // npoints is needed to bound the calculation
    const int ni = iline.n;

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
                append_solid_points<I>(info, m.solid_coord, ni - 1);
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

geometry::geometry(std::span<const shape> shapes, const mesh& m, bool check_domain)
{
    std::array<umesh_line, 3> lines{m.line(0), m.line(1), m.line(2)};
    init_line<0>(shapes, lines, rx_, rx_m_);
    init_line<1>(shapes, lines, ry_, ry_m_);
    init_line<2>(shapes, lines, rz_, rz_m_);

    init_solid<0>(lines, rx_, sx_);
    init_solid<1>(lines, ry_, sy_);
    init_solid<2>(lines, rz_, sz_);

    // solid points are used as storage for data associated with
    // boundary points.  For now we just crash if there isn't enough.
    // A different strategy would be to "pad" the geometry with extra
    // solid points and require all allocations over the domain to
    // contain this extra padding.
    if (check_domain) {
        if (sx_.size() < rx_.size())
            throw std::runtime_error(
                "Not enough solid points in x to accomidate boundaries");
        if (sy_.size() < ry_.size())
            throw std::runtime_error(
                "Not enough solid points in x to accomidate boundaries");
        if (sz_.size() < rz_.size())
            throw std::runtime_error(
                "Not enough solid points in x to accomidate boundaries");
    }
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