#include "mesh.hpp"

#include <sol/sol.hpp>

#include <spdlog/sinks/basic_file_sink.h>

namespace ccs
{

namespace
{
// We need to return lines of boundary_info so the caller
// can build up the appropriate operators
// the lines will be of a few different types:
// [domain, domain]
// [domain, object]
// [object, domain]
// [onbject, object]
//
// As with the solid point identification algorithm, we do not
// properly handle the case of fully solid lines
template <int J, int K>
constexpr bool same_plane(const int3& x, const int3& y)
{
    return x[J] == y[J] && x[K] == y[K];
}

// constexpr auto offset = [](int3 n) {
//     return [n](int3 ijk) { return ijk[0] * n[1] * n[2] + ijk[1] * n[2] + ijk[2]; };
// };

template <auto I>
void init_line(std::vector<line>& v, int3 extents, std::span<const mesh_object_info> r)
{
    // early exit if we are building operators in this direction
    if (extents[I] == 1) return;

    constexpr auto S = index::dir<I>::slow;
    constexpr auto F = index::dir<I>::fast;
    // auto off = offset(extents);

    integer ns = extents[S];
    integer nf = extents[F];

    v.reserve(ns * nf + r.size());
    auto first = rs::begin(r);
    auto last = rs::end(r);

    int3 left{};
    int3 right{};
    for (integer s = 0; s < ns; s++) {
        left[S] = s;
        right[S] = s;
        for (integer f = 0; f < nf; f++) {
            left[F] = f;
            right[F] = f;

            left[I] = 0;
            right[I] = extents[I] - 1;

            std::optional<boundary> left_boundary = boundary{left, std::nullopt};

            while (first != last && same_plane<S, F>(left, first->solid_coord)) {
                if (first->ray_outside) {
                    // set the `right` point and add both to line
                    v.emplace_back(
                        // off(left_boundary->mesh_coordinate),
                        index::stride<I>(extents),
                        *left_boundary,
                        boundary{.mesh_coordinate = first->solid_coord,
                                 .object = object_boundary{
                                     first - rs::begin(r), first->shape_id, first->psi}});
                    // invalidate the boundary point to indicate it was consumed
                    left_boundary.reset();
                } else {
                    // set the left_boundary and allow the next loop to process
                    left_boundary =
                        boundary{.mesh_coordinate = first->solid_coord,
                                 .object = object_boundary{
                                     first - rs::begin(r), first->shape_id, first->psi}};
                }
                ++first;
            }

            // consume the left boundary
            if (left_boundary) {
                v.emplace_back(
                    // off(left_boundary->mesh_coordinate),
                    index::stride<I>(extents),
                    *left_boundary,
                    boundary{.mesh_coordinate = right, .object = std::nullopt});
            }
        }
    }
}

void init_slices(std::vector<index_slice>& fluid_slices,
                 std::span<const line> lines,
                 index_extents extents)
{
    for (auto&& [_, start, end] : lines) {
        auto i0 = start.object ? extents(start.mesh_coordinate) + 1
                               : extents(start.mesh_coordinate);
        auto i1 =
            end.object ? extents(end.mesh_coordinate) : extents(end.mesh_coordinate) + 1;

        if (fluid_slices.empty()) {
            fluid_slices.emplace_back(i0, i1);
        } else {
            auto& [i0_prev, i1_prev] = fluid_slices.back();
            // if this slice is contiguous with the next, make it all one slice
            if (i1_prev == i0) {
                i1_prev = i1;
            } else {
                fluid_slices.emplace_back(i0, i1);
            }
        }
    }
}
} // namespace

mesh::mesh(const index_extents& extents,
           const domain_extents& bounds,
           const logs& build_logger)
    : mesh{extents, bounds, std::vector<shape>{}, build_logger}
{
}

mesh::mesh(const index_extents& extents,
           const domain_extents& bounds,
           const std::vector<shape>& shapes,
           const logs& build_logger)
    : cart{extents.extents, bounds.min, bounds.max},
      geometry{shapes, cart},
      logger{build_logger, "geometry", "geometry.csv"},
      xmin{sel::xmin(extents)},
      xmax{sel::xmax(extents)},
      ymin{sel::ymin(extents)},
      ymax{sel::ymax(extents)},
      zmin{sel::zmin(extents)},
      zmax{sel::zmax(extents)},
      xyz{cart.domain(), geometry.domain()},
      vxyz{tuple{tuple{cart.domain(), geometry.domain()},
                 tuple{cart.domain(), geometry.domain()},
                 tuple{cart.domain(), geometry.domain()}}}

{
    init_line<0>(lines_[0], cart.extents(), geometry.R(0));
    init_line<1>(lines_[1], cart.extents(), geometry.R(1));
    init_line<2>(lines_[2], cart.extents(), geometry.R(2));

    // setup fluid selector
    int i = extents[2] > 1 ? 2 : extents[1] > 1 ? 1 : 0;
    init_slices(fluid_slices, lines_[i], extents);
    fluid = sel::multi_slice(fluid_slices);

    logger.set_pattern("%v");
    logger(spdlog::level::info, "Timestamp,direction,ic,psi,x,y,z,i,j,k");
    logger.set_pattern("%Y-%m-%d %H:%M:%S.%f,%v");

    for (int dir = 0; dir < 3; dir++)
        for (int i = 0; auto&& [psi, pos, n, ray_out, ijk, id] : R(dir))
            logger(spdlog::level::info,
                   "{},{},{},{},{}",
                   dir,
                   i++,
                   psi,
                   fmt::join(pos, ", "),
                   fmt::join(ijk, ", "));
}

bool mesh::dirichlet_line(const int3& start, int dir, const bcs::Grid& cart_bcs) const
{
    bool result = false;

    auto f = [&](int i) {
        return (cart_bcs[i].left == bcs::Dirichlet &&
                cart.on_boundary(i, false, start)) ||
               (cart_bcs[i].right == bcs::Dirichlet && cart.on_boundary(i, true, start));
    };

    for (int i = 0; i < dir; i++) { result = result || f(i); }
    for (int i = dir + 1; i < 3; i++) { result = result || f(i); }

    return result;
}

line mesh::interp_line(int dir, int3 pt) const
{
    // start with mesh boundaries and bring bounds closer if needed
    int3 l{pt};
    l[dir] = 0;
    int3 r{pt};
    r[dir] = extents()[dir] - 1;
    boundary start{l, std::nullopt};
    boundary end{r, std::nullopt};

    const auto [f, s] = index::dirs(dir);

    for (integer i = -1; auto&& [psi, pos, n, ray_out, ijk, id] : R(dir)) {
        ++i;

        if (ijk[s] > pt[s]) break;
        if (ijk[s] != pt[s] || ijk[f] != pt[f]) continue;

        // check left/right  boundary
        if (n[dir] >= 0.0 && ijk[dir] <= pt[dir]) {
            assert(n[dir] != 0.0);
            // use >= just in case the object has a solid_coord on a domain wall
            if (ijk[dir] >= start.mesh_coordinate[dir])
                start = boundary{ijk, object_boundary{i, id, psi}};

        } else if (n[dir] < 0.0 && ijk[dir] >= pt[dir]) {
            assert(n[dir] != 0.0);
            if (ijk[dir] <= end.mesh_coordinate[dir])
                end = boundary{ijk, object_boundary{i, id, psi}};
        }
    }

    // need to reexamine the case for interpolating when pt coincides with two solid
    // points.
    assert(start.mesh_coordinate != end.mesh_coordinate);

    return {stride(dir), start, end};
}

std::optional<mesh> mesh::from_lua(const sol::table& tbl, const logs& logger)
{
    auto m_opt = cartesian::from_lua(tbl, logger);
    if (!m_opt) return std::nullopt;
    auto&& [n, domain] = *m_opt;

    auto shapes_opt = object_geometry::from_lua(tbl, n, domain, logger);
    if (!shapes_opt) return std::nullopt;
    const auto& shapes = *shapes_opt;

    return mesh{n, domain, shapes, logger};
    // return std::nullopt;
}

} // namespace ccs
