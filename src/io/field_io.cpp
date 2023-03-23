#include "field_io.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>

#include "mesh/cartesian.hpp"
#include "temporal/step_controller.hpp"

#include <fmt/core.h>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/transform.hpp>

#include <sol/sol.hpp>

using namespace std::literals;

namespace ccs
{

namespace fs = std::filesystem;

field_io::field_io(xdmf&& xdmf_w,
                   field_data&& field_data_w,
                   d_interval&& dump_interval,
                   std::string&& io_dir,
                   int suffix_length,
                   const logs& build_logger)
    : xdmf_w{MOVE(xdmf_w)},
      field_data_w{MOVE(field_data_w)},
      dump_interval{MOVE(dump_interval)},
      io_dir{MOVE(io_dir)},
      suffix_length{suffix_length},
      logger{build_logger, "field_io"}
{
}
bool field_io::write(std::span<const std::string> names,
                     field_view f,
                     const step_controller& step,
                     real dt,
                     tuple<std::span<const mesh_object_info>,
                           std::span<const mesh_object_info>,
                           std::span<const mesh_object_info>> r)
{
    if (!dump_interval(step, dt)) return false;

    fs::path io{io_dir};
    if ((int)step == 0) fs::create_directories(io);

    int n = dump_interval.current_dump();

    // prepare data for xdmf writer
    auto xmf_file_names = names | vs::transform([n, l = suffix_length](auto&& name) {
                              return fmt::format("{}.{:0{}d}", name, n, l);
                          }) |
                          rs::to<std::vector<std::string>>();

    xdmf_w.write(n, step, names, xmf_file_names, r, logger);

    if (n == 0) {
        auto file_names = std::vector<std::string>{io / "rx", io / "ry", io / "rz"};
        field_data_w.write_geom(file_names, r);
    }

    auto data_file_names = xmf_file_names |
                           vs::transform([io](auto&& name) { return io / name; }) |
                           rs::to<std::vector<std::string>>();

    field_data_w.write(f, data_file_names);

    ++dump_interval;
    return true;
}

std::optional<field_io> field_io::from_lua(const sol::table& tbl, const logs& logger)
{
    auto cart_opt = cartesian::from_lua(tbl);
    if (!cart_opt) return std::nullopt;

    auto io = tbl["io"];
    if (!io.valid()) return field_io{};

    sol::optional<int> write_every_step = io["write_every_step"];
    sol::optional<real> write_every_time = io["write_every_time"];
    std::string dir = io["dir"].get_or("io"s);
    int len = io["suffix_length"].get_or(6);
    std::string xmf_base = io["xdmf_filename"].get_or("view.xmf"s);

    if (write_every_step) {
        logger(
            spdlog::level::info, "field io will write every {} steps", *write_every_step);
    } else if (write_every_time) {
        logger(spdlog::level::info,
               "field io will write every {} time interval",
               *write_every_time);
    } else {
        logger(spdlog::level::info, "field io will not output data");
    }

    auto d = fs::path{dir};

    auto&& [ix, dom] = *cart_opt;

    auto xdmf_w = xdmf{d / xmf_base, ix, dom};
    auto data_w = field_data{ix};
    auto step = write_every_step ? interval<int>{*write_every_step} : interval<int>{};
    auto time = write_every_time ? interval<real>{*write_every_time} : interval<real>{};

    return field_io{
        MOVE(xdmf_w), MOVE(data_w), d_interval{step, time}, MOVE(dir), len, logger};
}
} // namespace ccs
