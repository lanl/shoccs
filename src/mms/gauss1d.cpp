#include "manufactured_solutions.hpp"
#include <cmath>
#include <vector>

#include "gauss.hpp"

namespace ccs
{

struct gauss1d : detail::gauss {
    using detail::gauss::gauss;

    real operator()(real time, const real3& loc) const
    {
        real sol = 0;
        const auto [x, y, z] = loc;
        for (int i = 0; i < static_cast<int>(center.size()); ++i) {
            sol += std::exp(-0.5 * std::pow((x - center[i][0]), 2) *
                            std::pow(variance[i][0], -2)) *
                   amplitude[i] * std::cos(time * frequency[i]);
        }
        return sol;
    }

    real ddt(real time, const real3& loc) const
    {
        real sol = 0;
        const auto [x, y, z] = loc;
        for (int i = 0; i < static_cast<int>(center.size()); ++i) {
            sol += -std::exp(-0.5 * std::pow((x - center[i][0]), 2) *
                             std::pow(variance[i][0], -2)) *
                   amplitude[i] * frequency[i] * std::sin(time * frequency[i]);
        }
        return sol;
    }

    real3 gradient(real time, const real3& loc) const
    {
        real3 sol{};
        const auto [x, y, z] = loc;
        for (int i = 0; i < static_cast<int>(center.size()); ++i) {
            sol[0] += -std::exp(-0.5 * std::pow((x - center[i][0]), 2) *
                                std::pow(variance[i][0], -2)) *
                      amplitude[i] * std::cos(time * frequency[i]) * (x - center[i][0]) *
                      std::pow(variance[i][0], -2);
        }
        return sol;
    }

    // This is a scalar field
    real divergence(real time, const real3& loc) const { return 0.0; }

    real laplacian(real time, const real3& loc) const
    {
        real sol = 0;
        const auto [x, y, z] = loc;
        for (int i = 0; i < static_cast<int>(center.size()); ++i) {
            sol += amplitude[i] * std::cos(time * frequency[i]) *
                   (std::exp(-0.5 * std::pow((x - center[i][0]), 2) *
                             std::pow(variance[i][0], -2)) *
                        std::pow((x - center[i][0]), 2) * std::pow(variance[i][0], -4) -
                    std::exp(-0.5 * std::pow((x - center[i][0]), 2) *
                             std::pow(variance[i][0], -2)) *
                        std::pow(variance[i][0], -2));
        }
        return sol;
    }
};

manufactured_solution build_ms_gauss1d(std::span<const real3> center,
                                       std::span<const real3> variance,
                                       std::span<const real> amplitude,
                                       std::span<const real> frequency)
{
    return {gauss1d{center, variance, amplitude, frequency}};
}

} // namespace ccs
