#include "laplacian.hpp"

#include "io/logging.hpp"

#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <fmt/ranges.h>
#include <string>
#include <vector>

namespace ccs
{

laplacian::laplacian(const mesh& m,
                     const stencil& st,
                     const bcs::Grid& grid_bcs,
                     const bcs::Object& obj_bcs,
                     const logs& build_logger)

{
    logs logger{build_logger, "laplacian", "laplacian.csv"};
    logger.set_pattern("%v");
    auto st_info = st.query_max();
    std::vector<std::string> hdr(st_info.t - 1, "wall,psi");
    logger(spdlog::level::info,
           "timestamp,deriv,interp_dir,ic,y,psi,{}",
           fmt::join(hdr, ","));
    logger.set_pattern("%Y-%m-%d %H:%M:%S.%f,%v");

    dx = derivative{0, m, st, grid_bcs, obj_bcs, logger};
    dy = derivative{1, m, st, grid_bcs, obj_bcs, logger};
    dz = derivative{2, m, st, grid_bcs, obj_bcs, logger};
    ex = m.extents();
}

// when there are no neumann conditions in the problem
std::function<void(scalar_span)> laplacian::operator()(scalar_view u) const
{
    return [this, u](scalar_span du) {
        Kokkos::Profiling::ScopedRegion region("laplacian::operator()");
        du = 0;
        if (ex[0] > 1) dx(u, du, plus_eq);
        if (ex[1] > 1) dy(u, du, plus_eq);
        if (ex[2] > 1) dz(u, du, plus_eq);
    };
}

std::function<void(scalar_span)> laplacian::operator()(scalar_view u,
                                                       scalar_view nu) const
{
    return [this, u, nu](scalar_span du) {
        Kokkos::Profiling::ScopedRegion region("laplacian::operator()");
        du = 0;
        // accumulate results into du
        if (ex[0] > 1) dx(u, nu, du, plus_eq);
        if (ex[1] > 1) dy(u, nu, du, plus_eq);
        if (ex[2] > 1) dz(u, nu, du, plus_eq);
    };
}
void laplacian::build_graph(scalar_view u, scalar_span du)
{
    graph_ = Kokkos::Experimental::create_graph<execution_space>(
        [&](auto root) { add_graph_nodes(root, u, du); });

    graph_->instantiate();
}

void laplacian::build_graph(scalar_view u, scalar_view nu, scalar_span du)
{
    graph_ = Kokkos::Experimental::create_graph<execution_space>(
        [&](auto root) { add_graph_nodes(root, u, nu, du); });

    graph_->instantiate();
}

void laplacian::submit_graph()
{
    Kokkos::Profiling::ScopedRegion region("laplacian::submit_graph()");
    graph_->submit();
    Kokkos::fence("laplacian::submit_graph() complete");
}

} // namespace ccs
