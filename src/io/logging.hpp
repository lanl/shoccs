#pragma once

#include <fmt/core.h>
#include <spdlog/spdlog.h>

namespace ccs
{
struct logs {
private:
    bool enable;

    mutable std::shared_ptr<spdlog::logger> logger;
    std::string logging_dir;

public:
    logs() = default;

    logs(const std::string& logging_dir, bool enable, const std::string& logger_name);
    logs(bool enable, const std::string& logger_name) : logs{"", enable, logger_name} {}

    logs(const logs& logger, const std::string& logger_name);

    logs(const logs& logger,
         const std::string& logger_name,
         const std::string& file_name);

    void set_pattern(const std::string& pat);

    operator bool() const { return enable; }

    void operator()(spdlog::level::level_enum lvl, const std::string& msg) const
    {
        if (*this) logger->log(lvl, msg);
    }

    template <typename... Args>
    void operator()(spdlog::level::level_enum lvl,
                    fmt::format_string<Args...> fmt_str,
                    Args&&... args) const
    {
        if (*this) logger->log(lvl, fmt_str, std::forward<Args>(args)...);
    }
};

} // namespace ccs
