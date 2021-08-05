#include "gradient.hpp"

#include <spdlog/sinks/basic_file_sink.h>

namespace ccs
{
gradient::gradient(const mesh& m,
                   const stencil& st,
                   const bcs::Grid& grid_bcs,
                   const bcs::Object& obj_bcs)
    : dx{0, m, st, grid_bcs, obj_bcs},
      dy{1, m, st, grid_bcs, obj_bcs},
      dz{2, m, st, grid_bcs, obj_bcs},
      ex{m.extents()}
{
}

gradient::gradient(const mesh& m,
                   const stencil& st,
                   const bcs::Grid& grid_bcs,
                   const bcs::Object& obj_bcs,
                   const std::string& logger_filename)
{
    auto sink =
        std::make_shared<spdlog::sinks::basic_file_sink_st>(logger_filename, true);
    logger = std::make_shared<spdlog::logger>("gradient", sink);
    logger->set_pattern("%v");
    auto st_info = st.query_max();
    logger->info(fmt::format("timestamp,deriv,interp_dir,ic,y,psi,{}",
                             fmt::join(vs::repeat_n("wall,psi", st_info.t - 1), ",")));
    logger->set_pattern("%Y-%m-%d %H:%M:%S.%f,%v");

    dx = derivative{0, m, st, grid_bcs, obj_bcs, logger};
    dy = derivative{1, m, st, grid_bcs, obj_bcs, logger};
    dz = derivative{2, m, st, grid_bcs, obj_bcs, logger};
    ex = m.extents();
}

std::function<void(vector_span)> gradient::operator()(scalar_view u) const
{
    return std::function<void(vector_span)>{[this, u](vector_span du) {
        du = 0;
        if (ex[0] > 1) dx(u, get<vi::X>(du));
        if (ex[1] > 1) dy(u, get<vi::Y>(du));
        if (ex[2] > 1) dz(u, get<vi::Z>(du));
    }};
}
} // namespace ccs
