#include "FieldIO.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>

#include "step_controller.hpp"

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/transform.hpp>

namespace ccs
{
namespace fs = std::filesystem;

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
void FieldIO::write(const field&, const step_controller&, real) {}
} // namespace ccs
