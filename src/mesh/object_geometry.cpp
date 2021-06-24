#include "object_geometry.hpp"
#include "indexing.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

#include <sol/sol.hpp>
#include <spdlog/spdlog.h>
//
// Somewhere in here is where we should form the vector of pairs of wall info
// that will be used to characterize the domain and build discretization stencils
//

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

object_geometry::object_geometry(std::span<const shape> shapes, const cartesian& m)
{
    std::array<umesh_line, 3> lines{m.line(0), m.line(1), m.line(2)};
    init_line<0>(shapes, lines, rx_, rx_m_);
    init_line<1>(shapes, lines, ry_, ry_m_);
    init_line<2>(shapes, lines, rz_, rz_m_);

    init_solid<0>(lines, rx_, sx_);
    init_solid<1>(lines, ry_, sy_);
    init_solid<2>(lines, rz_, sz_);
}

std::span<const mesh_object_info> object_geometry::Rx() const { return rx_; }

std::span<const mesh_object_info> object_geometry::Rx(int shape_id) const
{
    return rx_m_[shape_id];
}

std::span<const mesh_object_info> object_geometry::Ry() const { return ry_; }

std::span<const mesh_object_info> object_geometry::Ry(int shape_id) const
{
    return ry_m_[shape_id];
}

std::span<const mesh_object_info> object_geometry::Rz() const { return rz_; }

std::span<const mesh_object_info> object_geometry::Rz(int shape_id) const
{
    return rz_m_[shape_id];
}

std::span<const int3> object_geometry::Sx() const { return sx_; }
std::span<const int3> object_geometry::Sy() const { return sy_; }
std::span<const int3> object_geometry::Sz() const { return sz_; }

std::optional<std::vector<shape>> object_geometry::from_lua(const sol::table& tbl)
{
    auto t = tbl["shapes"];
    if (!t.valid()) {
        spdlog::info("No cut-cell shapes specified");
        return std::vector<shape>{};
    }

    std::vector<shape> s{};
    for (int i = 1; t[i].valid(); i++) {
        auto type = t[i]["type"].get_or(std::string{});

        if (type == "sphere") {

            real3 center{t[i]["center"][1].get_or(0.0),
                         t[i]["center"][2].get_or(0.0),
                         t[i]["center"][3].get_or(0.0)};
            real radius = t[i]["radius"].get_or(0.0);
            s.push_back(make_sphere(i - 1, center, radius));

            spdlog::info("found sphere of radius {} and center {}",
                         radius,
                         fmt::join(center, ", "));

        } else if (type == "yz_rect") {

            real3 lc{t[i]["lower_corner"][1].get_or(0.0),
                     t[i]["lower_corner"][2].get_or(0.0),
                     t[i]["lower_corner"][3].get_or(0.0)};
            real3 uc{t[i]["upper_corner"][1].get_or(0.0),
                     t[i]["upper_corner"][2].get_or(0.0),
                     t[i]["upper_corner"][3].get_or(0.0)};
            real n = t[i]["normal"].get_or(1.0);
            s.push_back(make_yz_rect(i - 1, lc, uc, n));

        } else {
            spdlog::error("shape type must be one of: sphere, yz_rect ...");
            return std::nullopt;
        }
    }

    return s;
}

} // namespace ccs
