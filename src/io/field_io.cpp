#include "field_io.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>

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
                   int suffix_length)
    : xdmf_w{MOVE(xdmf_w)},
      field_data_w{MOVE(field_data_w)},
      dump_interval{MOVE(dump_interval)},
      io_dir{MOVE(io_dir)},
      suffix_length{suffix_length}
{
}

// const fs::path io_dir{"io"};
#if 0
field_io::field_io(xdmf&& xwriter,
                   field_data&& fwriter,
                   interval<int> step_interval,
                   interval<double> time_interval,
                   std::string io_dir)
    : xdmf_writer{std::move(xwriter)},
      field_data_writer{std::move(fwriter)},
      step_interval{step_interval},
      time_interval{time_interval},
      io_dir{io_dir}
{
    // if both intervals are huge we won't be writing anything
    wont_write = !(step_interval || time_interval);
}

void field_io::add(const std::string& name, const real* data)
{
    if (wont_write) return;

    v.emplace_back(name, data);
}
void field_io::write(int step, real time, real dt)
{
    if (wont_write) return;

    fs::path io{io_dir};

    if (step == 0) fs::create_directories(io);

    auto step_ready = step_interval(step);
    auto time_ready = time_interval(time, dt / 2.0);
    // check to see if writing is requested
    if (!(step_ready || time_ready || step == 0)) return;

    // prepare the file suffix
    const std::string suffix = [](int len, int step) {
        std::ostringstream os{};
        os << "." << std::setw(len) << std::setfill('0') << step;
        return os.str();
    }(suffix_length, current_step);

    // xdmf
    std::vector<std::pair<std::string, std::string>> vars_and_files =
        v | vs::transform([&suffix](const auto& item) {
            const auto& name = std::get<0>(item);
            // be explicit about std::string so we dont create references to temps
            return std::pair<std::string, std::string>{name, name + suffix};
        }) |
        rs::to<std::vector>();

    if (step > 0) {
        std::fstream stream{io / "view.xmf"};
        xdmf_writer.readwrite(stream, current_step, time, vars_and_files);
    } else {
        std::ofstream stream{io / "view.xmf"};
        xdmf_writer.write(stream, vars_and_files);
    }

    // binary io
    for (const auto& [name, data] : v) {
        // construct filename
        std::ofstream stream{io / (name + suffix)};
        field_data_writer.write(stream, data);
    }

    // finally, update intervals as needed
    if (step_ready) ++step_interval;
    if (time_ready) ++time_interval;

    ++current_step;
}
#endif

bool field_io::write(std::span<const std::string> names,
                     const field& f,
                     const step_controller& step,
                     real dt)
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

    xdmf_w.write(n, step, names, xmf_file_names);

    auto data_file_names = xmf_file_names |
                           vs::transform([io](auto&& name) { return io / name; }) |
                           rs::to<std::vector<std::string>>();
    field_data_w.write(f, data_file_names);

    ++dump_interval;
    return true;
}

std::optional<field_io>
field_io::from_lua(const sol::table& tbl, index_extents ix, const domain_extents& dx)
{
    auto io = tbl["io"];
    if (!io.valid()) return field_io{};

    sol::optional<int> write_every_step = io["write_every_step"];
    sol::optional<real> write_every_time = io["write_every_step"];
    std::string dir = io["dir"].get_or("io"s);
    int len = io["suffix_length"].get_or(6);
    std::string xmf_base = io["xdmf_filename"].get_or("view.xmf"s);

    auto d = fs::path{dir};

    auto xdmf_w = xdmf{d / xmf_base, ix, dx};
    auto data_w = field_data{ix};
    auto step = write_every_step ? interval<int>{*write_every_step} : interval<int>{};
    auto time = write_every_time ? interval<real>{*write_every_time} : interval<real>{};

    return field_io{MOVE(xdmf_w), MOVE(data_w), d_interval{step, time}, MOVE(dir), len};
}
} // namespace ccs
