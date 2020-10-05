#pragma once

#include <fstream>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pdg
{
class logger
{
        std::ofstream stream_;

    public:
        logger() = default;

        logger(std::string filename, const std::vector<std::string>& header)
            : stream_{filename}
        {
                auto first = header.begin();
                auto last = header.end();
                if (first == last)
                        return;

                stream_ << *first++;

                while (first != last) { stream_ << ", " << *first++; }
                stream_ << "\n";
        }

        template <typename U, typename... V>
        void write(const U& u, const V&... v)
        {
                // variadic write of csv data
                stream_ << u;

                if (sizeof...(V) == 0) {
                        stream_ << "\n";
                        return;
                }

                ((stream_ << "," << v), ...);
                stream_ << "\n";
        }

        template <typename T>
        void write_vector(const std::vector<T>& v)
        {
                auto first = v.begin();
                auto last = v.end();

                while (first != last) { stream_ << *first++ << "\n"; }
        }

        void flush() { stream_.flush(); }
};

using loggers = std::unordered_map<std::string, std::optional<logger>>;
} // namespace pdg
