#include "system.hpp"

#include <cassert>

namespace pdg
{

system::system(cart_mesh&& cart, mesh&& cut_mesh)
    : cart{std::move(cart)}, cut_mesh{std::move(cut_mesh)}
{
}

double system::timestep_size(double cfl, double max_step_size) const
{
        return std::min(this->system_timestep_size(cfl), max_step_size);
}

void system::prestep() { u0 = u; }

void system::update(std::vector<double>& rhs, double dtf)
{
        for (int i = 0, end = u.size(); i < end; ++i) u[i] = u0[i] + dtf * rhs[i];

        // This is where we could apply boundary conditions..., but they get applied in ()
}

std::vector<double> system::operator()(double time)
{
        std::vector<double> rhs(u.size());
        (*this)(rhs, time);
        return rhs;
}

bool system::valid(const system_stats& stats)
{
        return stats.u_min > -100.0 && stats.u_max < 100.0 && stats.Linf < 100.0;
}

void system::log(std::optional<logger>& lg,
                 const system_stats& sstats,
                 int step,
                 double time) const
{
        if (lg)
                (*lg).write(time, step, sstats.u_max, sstats.u_min, sstats.Linf);
}

std::vector<double> system::current_solution() const { return u; }
std::vector<double> system::current_error() const { return error; }
} // namespace pdg
