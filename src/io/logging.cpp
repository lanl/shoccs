#include "logging.hpp"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <filesystem>

namespace fs = std::filesystem;

namespace ccs
{
logs::logs(const std::string& logging_dir, bool enable, const std::string& name)
    : enable{enable}, logging_dir{logging_dir}
{
    if (enable) {
        auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_st>();
        logger = std::make_shared<spdlog::logger>(name, sink);
    }
}

logs::logs(const logs& other, const std::string& logger_name) : enable{other}
{
    if (enable) {
        auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_st>();
        logger = std::make_shared<spdlog::logger>(logger_name, sink);
    }
}

logs::logs(const logs& other,
           const std::string& logger_name,
           const std::string& file_name)
    : enable(other), logging_dir(other.logging_dir)
{
    if (enable) {
        auto sink = std::make_shared<spdlog::sinks::basic_file_sink_st>(
            fs::path{logging_dir} / file_name, true);
        logger = std::make_shared<spdlog::logger>(logger_name, sink);
    }
}

void logs::set_pattern(const std::string& pat)
{
    if (enable) { logger->set_pattern(pat); }
}
} // namespace ccs
