#include "gradient.hpp"

#include "io/logging.hpp"

#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <fmt/ranges.h>
#include <string>
#include <vector>

namespace ccs
{
gradient::gradient(const mesh& m,
                   const stencil& st,
                   const bcs::Grid& grid_bcs,
                   const bcs::Object& obj_bcs,
                   const logs& build_logger)
{
    logs logger{build_logger, "gradient", "gradient.csv"};
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

std::function<void(scalar_span, scalar_span, scalar_span)>
gradient::operator()(scalar_view u) const
{
    return [this, u](scalar_span du_x, scalar_span du_y, scalar_span du_z) {
        Kokkos::Profiling::ScopedRegion region("gradient::operator()");
        du_x = 0;
        du_y = 0;
        du_z = 0;
        if (ex[0] > 1) dx(u, du_x);
        if (ex[1] > 1) dy(u, du_y);
        if (ex[2] > 1) dz(u, du_z);
    };
}
} // namespace ccs
