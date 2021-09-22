#include "logging.hpp"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace ccs
{
logs::logs(bool enable, const std::string& name) : enable{enable}
{
    if (enable) {
        auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_st>();
        logger = std::make_shared<spdlog::logger>(name, sink);
    }
}

logs::logs(bool enable, const std::string& logger_name, const std::string& file_name)
    : enable{enable}
{
    if (enable) {
        auto sink = std::make_shared<spdlog::sinks::basic_file_sink_st>(file_name, true);
        logger = std::make_shared<spdlog::logger>(logger_name, sink);
    }
}

void logs::set_pattern(const std::string& pat)
{
    if (enable) { logger->set_pattern(pat); }
}
} // namespace ccs
