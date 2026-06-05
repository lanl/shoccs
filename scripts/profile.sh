#!/bin/bash
# Run SHOCCS with Kokkos profiling tools enabled.
#
# Usage:
#   ./scripts/profile.sh config.lua [extra-args...]
#
# This loads the SimpleKernelTimer tool from kokkos-tools so that every
# Kokkos::parallel_for / parallel_reduce and every ScopedRegion is timed.
# After the run, kp_reader decodes the binary .dat files into a human-
# readable summary.
#
# Prerequisites:
#   kokkos-tools must be installed (listed in .devcontainer/spack.yaml).
#
# To use a different tool (e.g. space-time stack):
#   KOKKOS_TOOLS_LIBS=$(spack location -i kokkos-tools)/lib/libkp_space_time_stack.so \
#       ./build/src/app/shoccs config.lua

set -euo pipefail

KT_PREFIX="${KOKKOS_TOOLS_PREFIX:-$(spack location -i kokkos-tools 2>/dev/null || echo "")}"

if [[ -z "$KT_PREFIX" ]]; then
    echo "Error: kokkos-tools not found. Install via spack or set KOKKOS_TOOLS_PREFIX." >&2
    exit 1
fi

KT_LIB="${KT_PREFIX}/lib/libkp_kernel_timer.so"
if [[ ! -f "$KT_LIB" ]]; then
    echo "Error: $KT_LIB not found." >&2
    exit 1
fi

export KOKKOS_TOOLS_LIBS="$KT_LIB"
echo "Profiling with: $KT_LIB"

./build/src/app/shoccs "$@"

# Decode binary profiling output
if command -v kp_reader &>/dev/null; then
    kp_reader *.dat
else
    echo "kp_reader not found in PATH; .dat files written to $(pwd)"
fi
