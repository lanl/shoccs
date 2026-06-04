#pragma once

#include <Kokkos_Core.hpp>

#include "shoccs_config.hpp"

namespace ccs
{

using execution_space = Kokkos::DefaultHostExecutionSpace;
using memory_space = typename execution_space::memory_space;

template <typename T>
using device_view = Kokkos::View<T, memory_space>;

} // namespace ccs
