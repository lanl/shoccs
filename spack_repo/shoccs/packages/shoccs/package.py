# Copyright (c) 2022, Triad National Security, LLC. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Spack 1.0+ / Package API v2: build-system base classes now live in the
# (separately versioned) builtin package repository, not in core spack. The
# directives (version, variant, depends_on, ...) still come from spack.package.
from spack_repo.builtin.build_systems.cmake import CMakePackage

from spack.package import *


class Shoccs(CMakePackage):
    """SHOCCS (Stable High-Order Cut-Cell Solver): a C++20 Cartesian cut-cell
    solver for time-dependent PDEs (heat equation, scalar wave, Euler
    equations). It uses high-order finite-difference operators on structured
    grids with embedded boundaries and is parallelized with Kokkos."""

    # TODO(maintainer): confirm canonical repo URL (origin remote is
    # github.com/pbrady/shoccs; upstream is github.com/lanl/shoccs).
    homepage = "https://github.com/lanl/shoccs"
    git = "https://github.com/lanl/shoccs.git"

    maintainers("pbrady")

    license("BSD-3-Clause")

    # Development version tracked from the default branch. The devcontainer and
    # `spack develop` workflows build from a live checkout, so no sha256 is
    # needed here.
    version("main", branch="main")
    # TODO(maintainer): add tagged releases as they are cut, e.g.:
    #   version("1.0.0", sha256="<spack checksum shoccs 1.0.0>")

    # ------------------------------------------------------------------ #
    # Build options (mapped to the project's CMake cache options below)  #
    # ------------------------------------------------------------------ #
    variant("tests", default=True, description="Build the Catch2 unit tests")
    variant(
        "benchmarks",
        default=False,
        description="Build the Google Benchmark microbenchmark suite",
    )
    # Off by default so the standard build stays fully pinned/offline-capable
    # (kokkos-tools has only an unpinned `develop` branch — see below). The
    # devcontainer turns this on explicitly.
    variant(
        "profiling",
        default=False,
        description="Install kokkos-tools for runtime Kokkos profiling "
        "(loaded via KOKKOS_TOOLS_LIBS)",
    )

    # ------------------------------------------------------------------ #
    # Language (compiler) dependencies                                   #
    # Spack 1.0 models compilers as real nodes in the DAG, so a package  #
    # must declare the language virtuals it compiles. SHOCCS is          #
    # project(shoccs LANGUAGES CXX); C is pulled in by CMake probing and #
    # several transitive deps, so both are declared.                     #
    # ------------------------------------------------------------------ #
    depends_on("c", type="build")
    depends_on("cxx", type="build")

    # ------------------------------------------------------------------ #
    # Build tools                                                        #
    # ------------------------------------------------------------------ #
    depends_on("cmake@3.22:", type="build")
    depends_on("ninja", type="build")

    # ------------------------------------------------------------------ #
    # Link / run libraries (default dependency type is build+link)       #
    # ------------------------------------------------------------------ #
    depends_on("lua")
    depends_on("lua-sol2")
    depends_on("fmt@8:")
    depends_on("pugixml@1.10:")
    depends_on("spdlog@1.9:")
    depends_on("cxxopts")
    # SHOCCS only uses Boost header-only (Boost.MP11). The spack boost recipe
    # has no header_only variant; instead every compiled-library variant
    # defaults off, so a plain `boost` spec yields a cheap header-only install
    # (Boost is configured with --with-libraries=headers).
    depends_on("boost")
    # C++ LAPACK/BLAS wrapper. Pulls the blas/lapack virtuals; the concrete
    # provider (e.g. openblas) is chosen by site/environment config.
    depends_on("lapackpp")

    # Host-only Kokkos. The serial and openmp backends both default to off in
    # the current recipe, so +serial must be requested explicitly. cxxstd=20 is
    # the default (and only sensible value) for kokkos@5:, stated for clarity.
    # The solver's create_graph call sites use the Kokkos 5.1 Graph API
    # (one-arg create_graph<execution_space>(closure), DeviceHandle era). Bound
    # to the validated 5.1.x series: 5.1.0 is the floor (the 5.0 two-arg form no
    # longer compiles) and the upper bound guards against a future 5.2/6.0
    # changing this still-Experimental overload — widen only after re-validating
    # the call sites. As of June 2026 kokkos 5.1.x lives only on the
    # spack-packages `develop` branch (newest tag v2026.03.0 caps at 5.0.2); the
    # devcontainer pins a develop commit, native users point their builtin there.
    depends_on("kokkos@5.1.0:5.1 +serial +openmp cxxstd=20")
    # Enable with +profiling. kokkos-tools currently ships only a `develop`
    # branch version (no tagged release), so this resolves to @develop and needs
    # git access at build time; it is therefore left off by default.
    depends_on("kokkos-tools", type=("build", "run"), when="+profiling")

    # ------------------------------------------------------------------ #
    # Test- / benchmark-only dependencies                                #
    # ------------------------------------------------------------------ #
    depends_on("catch2@3:", when="+tests")
    depends_on("benchmark", when="+benchmarks")

    def cmake_args(self):
        return [
            self.define_from_variant("BUILD_TESTING", "tests"),
            self.define_from_variant("BUILD_BENCHMARKS", "benchmarks"),
        ]
