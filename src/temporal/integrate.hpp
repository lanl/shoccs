#pragma once
#include "../systems/system.hpp"
#include <memory>

namespace pdg
{
class integrate
{
    public:
        virtual void step(system& sys, double time, double dt) = 0;

        virtual ~integrate() = default;
};

std::unique_ptr<integrate> build_rk4(int_t rhs_sz);
std::unique_ptr<integrate> build_euler(int_t rhs_sz);
} // namespace pdg
