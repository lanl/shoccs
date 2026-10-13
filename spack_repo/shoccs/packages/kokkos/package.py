# Copyright (c) 2022, Triad National Security, LLC. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Thin override of the builtin kokkos recipe for native macOS builds.
#
# Apple's clang ships no OpenMP runtime, and the builtin recipe does not ask
# for one, so `kokkos +openmp %apple-clang` fails inside Kokkos's
# find_package(OpenMP). Depending on llvm-openmp (libomp) the way blaspp and
# other builtin recipes do puts omp.h and libomp on the compiler wrapper's
# search paths and on CMAKE_PREFIX_PATH, which is all FindOpenMP needs. The
# directive is inert for gcc, so the devcontainer build is unchanged.
#
# The subclass inherits every version, variant, patch and conflict of the
# builtin package; this repo precedes `builtin` in the environment's repos
# list, so plain `kokkos` resolves here.
#
# Drop this override once spack-packages carries the same dependency (a
# one-line patch is proposed upstream). See docs/reference/macos-host-build.md.
from spack_repo.builtin.packages.kokkos.package import Kokkos as BuiltinKokkos

from spack.package import *


class Kokkos(BuiltinKokkos):
    depends_on("llvm-openmp", when="+openmp %apple-clang")
