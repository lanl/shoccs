>  ⚠️ HISTORICAL PLANNING DOC (pre-migration). The range-v3 -> Kokkos migration described here as future work is COMPLETE (2026-03). The subsystem decomposition and PDE/method-of-lines overview are still useful, but the range-v3 code examples, the migration roadmap (§6/§8/§9), `cc_elliptic` (never created), and several struct details (e.g. mesh_object_info) are obsolete. For current state see docs/CAPABILITY_AUDIT.md and docs/reference/. NOTE: operators use STRONG BC enforcement, not SBP-SAT despite §4.2's label.

# SHOCCS Architecture Specification & Kokkos Migration Guide

**Stable High-Order Cut-Cell Solver — Complete Codebase Analysis**
**Date:** 2026-03-09
**Purpose:** Comprehensive architectural decomposition for migrating from range-v3 to Kokkos

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Overview](#2-project-overview)
3. [What SHOCCS Solves](#3-what-shoccs-solves)
4. [How SHOCCS Solves Systems of Equations](#4-how-shoccs-solves-systems-of-equations)
5. [Subsystem Architecture](#5-subsystem-architecture)
   - 5.1 [Mesh & Geometry](#51-mesh--geometry)
   - 5.2 [Indexing System](#52-indexing-system)
   - 5.3 [Fields Subsystem](#53-fields-subsystem)
   - 5.4 [Stencils Subsystem](#54-stencils-subsystem)
   - 5.5 [Operators Subsystem](#55-operators-subsystem)
   - 5.6 [Matrices Subsystem](#56-matrices-subsystem)
   - 5.7 [Systems Subsystem (PDE Implementations)](#57-systems-subsystem)
   - 5.8 [Temporal Integration](#58-temporal-integration)
   - 5.9 [Simulation Orchestration](#59-simulation-orchestration)
   - 5.10 [I/O Subsystem](#510-io-subsystem)
   - 5.11 [MMS (Verification)](#511-mms-verification)
6. [Complete Range-v3 Usage Catalog](#6-complete-range-v3-usage-catalog)
7. [Data Flow: End-to-End](#7-data-flow-end-to-end)
8. [Kokkos Migration Strategy](#8-kokkos-migration-strategy)
9. [Migration Roadmap](#9-migration-roadmap)

---

## 1. Executive Summary

SHOCCS is a C++ Cartesian cut-cell solver for time-dependent PDEs (heat equation, scalar wave equation, inviscid vortex/Euler equations). It implements the numerical algorithm from Brady & Livescu (2020, J. Computational Physics). The code uses range-v3 extensively as a domain-specific language for expressing numerical operations on structured grids with embedded boundaries.

**Codebase stats:** 189 source files (hpp/cpp), ~16 subsystem directories, heavy use of C++20 concepts, range-v3 views/actions/algorithms, and coroutine generators.

**Key range-v3 dependencies:** 231 range-v3 header inclusions across the codebase, 6 custom view adaptors, 8+ view types, 11+ algorithms, coroutine generators for multi-dimensional iteration.

**Migration scope:** Replace all range-v3 lazy view composition with Kokkos parallel execution patterns while preserving the mathematical structure and correctness of the solver.

---

## 2. Project Overview

### 2.1 Dependencies

| Library | Purpose | Migration Impact |
|---------|---------|-----------------|
| range-v3 | View composition, lazy evaluation, algorithms | **REPLACED by Kokkos** |
| Lua / sol2 | Runtime configuration | No change |
| Catch2 | Unit testing | No change |
| fmt | String formatting | No change |
| pugixml | XDMF output | No change |
| spdlog | Logging | No change |
| cxxopts | CLI argument parsing | No change |

### 2.2 Directory Structure

```
src/
├── app/          # Main entry point (shoccs.cpp)
├── fields/       # Field data structures, selectors, tuple composition
├── geometry/     # (Placeholder — geometry logic lives in mesh/)
├── io/           # XDMF/binary output, logging, intervals
├── lib/          # Library entry point (run_from_sol)
├── matrices/     # CSR, dense, circulant, block, visitors
├── mesh/         # Cartesian mesh, cut-cell geometry, object intersection
├── mms/          # Method of manufactured solutions (Gaussian, Lua-defined)
├── operators/    # Gradient, Laplacian, divergence, derivative, boundaries
├── random/       # Random number generation
├── simulation/   # Builder pattern, simulation cycle
├── stencils/     # Finite difference stencil coefficients
├── systems/      # PDE implementations (heat, wave, vortex, eigenvalues)
├── temporal/     # Time integrators (Euler, RK4, step controller)
├── utils/        # Bounded types, extents
├── indexing.hpp  # Multi-dimensional index mapping
├── index_extents.hpp  # Grid extent management
├── index_view.hpp     # Index iteration views
├── types.hpp     # Type aliases (real, integer, int3, real3)
└── sentinels.hpp # Sentinel values
```

---

## 3. What SHOCCS Solves

### 3.1 Heat Equation (Parabolic PDE)

```
∂T/∂t = κ∇²T + S(t,x)
```

where κ is thermal diffusivity and S is a source term (from manufactured solutions).

**Configuration example** (`heat.lua`): 51×51 grid, diffusivity=1/30, Gaussian manufactured solution centered at (1,1) with a spherical embedded object using floating boundary conditions.

**Boundary conditions:** Dirichlet (prescribed temperature) and Neumann (prescribed flux) on domain boundaries; floating (interpolated) on cut-cell boundaries.

### 3.2 Scalar Wave Equation (Hyperbolic PDE)

```
∂u/∂t = -∇G · ∇u
```

where G is a vector field representing wave propagation direction. Simplifies to `u_t = -a·u_x - b·u_y - c·u_z`.

**Solution form:** `u(t,x) = sin(2π(‖x - center‖ - radius - t))` — a radially-symmetric traveling wave.

**Boundary conditions:** Dirichlet with prescribed exact solution values.

### 3.3 Inviscid Vortex / Euler Equations (Hyperbolic Conservation Laws)

```
∂ρ/∂t + ∇·(ρu) = 0          (mass)
∂(ρu)/∂t + ∇·(ρu⊗u + pI) = 0  (momentum)
∂(ρE)/∂t + ∇·((ρE + p)u) = 0   (energy)
```

**Status:** Partially implemented (core algorithms present but disabled via `#if 0` blocks). The inviscid vortex is a standard verification problem for Euler solvers.

### 3.4 Hyperbolic Eigenvalue Analysis (Utility)

Computes maximum eigenvalues of the discrete spatial operators for stability analysis. No time stepping — used to determine CFL limits for the hyperbolic schemes.

---

## 4. How SHOCCS Solves Systems of Equations

### 4.1 Method of Lines

SHOCCS uses the **method of lines** approach: spatial derivatives are discretized first (producing a large ODE system), then advanced in time with explicit integrators.

```
du/dt = f(u, t)    where f = spatial discretization operator
```

### 4.2 Spatial Discretization

Spatial derivatives use **summation-by-parts (SBP) finite difference operators** on the Cartesian grid, with special stencil modifications near embedded boundaries (cut cells).

**Operator assembly:**
1. Interior points: circulant (banded Toeplitz) matrices encoding high-order stencils
2. Near-boundary points: dense matrices with modified stencil coefficients
3. Cut-cell points: CSR matrices with geometry-dependent coefficients based on the `psi` parameter (normalized distance from surface to cell center)

**Composite structure:** Each 1D operator along a grid line is an `inner_block` = `[dense_left | circulant_interior | dense_right]`. Multiple lines compose into a `block` matrix. The full multi-dimensional operator combines these via the `discrete_operator` abstraction.

### 4.3 Time Integration

Two explicit integrators:

**Forward Euler (1st order):**
```
u^(n+1) = u^n + Δt · f(u^n, t^n)
```

**Classical RK4 (4th order):**
```
k1 = f(u^n, t^n)
k2 = f(u^n + (Δt/2)·k1, t^n + Δt/2)
k3 = f(u^n + (Δt/2)·k2, t^n + Δt/2)
k4 = f(u^n + Δt·k3, t^n + Δt)
u^(n+1) = u^n + (Δt/6)(k1 + 2k2 + 2k3 + k4)
```

### 4.4 Stability (CFL Conditions)

**Parabolic (heat):** `Δt = CFL_p · h_min² / (4κ)` — quadratic in mesh spacing.
**Hyperbolic (wave):** `Δt = CFL_h · h_min / c_max` — linear in mesh spacing.

Typical values: `CFL_p = 0.5`, `CFL_h = 0.8` for RK4.

### 4.5 Boundary Condition Enforcement

Applied in two phases per RHS evaluation:

1. **Pre-RHS:** Set Dirichlet values to exact/prescribed values at current time; store Neumann gradient data for operator use.
2. **Post-RHS:** Zero RHS at Dirichlet points (values enforced directly, not evolved).

Cut-cell boundaries use floating conditions: the solution is interpolated from interior values using geometry-dependent weights.

### 4.6 Verification via Method of Manufactured Solutions (MMS)

An exact solution Q(t,x) is chosen (e.g., Gaussian), and a source term S = ∂Q/∂t - L(Q) is computed analytically and added to the RHS. The discrete solution should converge to Q at the expected order of accuracy. Error is measured as L∞ norm over the fluid domain.

---

## 5. Subsystem Architecture

### 5.1 Mesh & Geometry

**Files:** `src/mesh/cartesian.hpp/cpp`, `mesh.hpp/cpp`, `mesh_types.hpp`, `mesh_view.hpp`, `object_geometry.hpp/cpp`, `rect.hpp/cpp`, `sphere.cpp`, `selections.hpp`, `shapes.hpp`

**Cartesian Grid Representation:**
```cpp
class cartesian : public index_extents {
    std::vector<real> x_, y_, z_;  // 1D coordinate arrays
    std::vector<real> h_;          // Grid spacings [hx, hy, hz]
    int dims_;                     // Number of active dimensions (1, 2, or 3)
};
```

- **Index extents** `n_[3]`: Number of points in each direction (inactive = 1)
- **Domain bounds** `min_[3]`, `max_[3]`: Physical domain extent
- **Coordinates:** 1D arrays distributed uniformly via `vs::linear_distribute`

**Cut-Cell Geometry:**

The `object_geometry` class performs ray-casting intersection of grid lines with embedded shapes (spheres, rectangles):

```cpp
struct mesh_object_info {
    real psi;           // Normalized distance: surface-to-center / h
    integer fluid_start; // Fluid region start index
    integer fluid_end;   // Fluid region end index
    bc_type bc;          // Boundary condition type
};
```

- **Rx, Ry, Rz arrays:** Per-direction intersection data for each grid line
- **Sx, Sy, Sz:** Solid interior point identification
- **psi parameter:** Critical for cut-cell stencil modification (0 = on surface, 0.5 = cell center)
- **Fluid slices:** Contiguous fluid regions separated by solid objects, stored as start/end pairs

**Mesh Selections:**

Selections partition the grid into regions for targeted operations:
- `sel::D` — Domain interior (non-ghost)
- `sel::R` — Ghost/reflection regions
- `sel::xR, yR, zR` — Direction-specific regions
- `m.fluid` — Interior fluid points
- `m.dirichlet(grid_bcs, object_bcs)` — Dirichlet boundary points
- `m.neumann<dir>(grid_bcs)` — Direction-specific Neumann boundaries
- `m.fluid_all(object_bcs)` — All non-Dirichlet fluid points

These are implemented as **range-v3 view factories** that return filtered index ranges.

**Range-v3 Usage:**
```cpp
// Grid coordinate generation
x_ = vs::linear_distribute(min_[0], max_[0], n_[0]) | rs::to<std::vector<real>>();

// Dimension detection
dims_ = rs::count_if(n_, [](auto n) { return !!(n - 1); });

// Grid spacing computation
rs::copy(vs::zip_with([](real mn, real mx, int n) { return (mx-mn)/(n-1); },
         min_, max_, n_), rs::begin(h_));

// Domain coordinate generation (Cartesian product)
auto domain() { return tuple{vs::cartesian_product(x(), y(), z())}; }
```

### 5.2 Indexing System

**Files:** `indexing.hpp`, `index_extents.hpp`, `index_view.hpp`

**Multi-dimensional Index Mapping:**

The indexing system maps 3D indices `(i,j,k)` to linear storage using a **direction-dependent convention**. For operations along direction `I`, the other two directions are classified as "slow" (large stride) and "fast" (small stride):

| Operation Dir | I (primary) | S (slow) | F (fast) | Linear = |
|--------------|-------------|----------|----------|----------|
| X (dir=0) | i | j (ny stride) | k (1 stride) | i·ny·nz + j·nz + k |
| Y (dir=1) | j | i (nx stride) | k (1 stride) | i·ny·nz + j·nz + k |
| Z (dir=2) | k | i (nx stride) | j (ny stride) | i·ny·nz + j·nz + k |

**Coroutine Generators for Iteration:**
```cpp
cppcoro::generator<int3> index_view(int3 extents) {
    for (int s = 0; s < extents[S]; s++)
        for (int f = 0; f < extents[F]; f++)
            for (int i = 0; i < extents[I]; i++)
                co_yield ijk;
}
```

This is a **critical migration point**: Kokkos replaces coroutine generators with `MDRangePolicy<Rank<3>>` parallel dispatch.

**Transpose Operations:** 9 transposition functions convert between direction-specific and canonical (x,y,z) ordering. These are tested exhaustively.

### 5.3 Fields Subsystem

**Files:** `src/fields/` (18 headers + 15 test files)

**Data Structure Hierarchy:**

```
field<S, V>
├── scalars: vector<scalar_real>
│   └── Each scalar: tuple<domain_vector, (Rx_vec, Ry_vec, Rz_vec)>
│       └── domain_vector: std::vector<real> (contiguous linear storage)
└── vectors: vector<vector_real>
    └── Each vector: tuple<scalar_x, scalar_y, scalar_z>
        └── Each component is a scalar_real
```

- **scalar_real** = a tuple of views into the flat storage (domain + ghost regions)
- **vector_real** = 3 scalar_real components
- **field** = collection of scalar and vector fields for a PDE system

**Custom View Adaptors (Most Complex range-v3 Usage):**

1. **`plane_view<0>`** (X-plane): Contiguous memory slice — maps to `Kokkos::subview`
2. **`plane_view<1>`** (Y-plane): Non-contiguous strided access — most complex, requires custom Kokkos kernel
3. **`plane_view<2>`** (Z-plane): Strided access — maps to `Kokkos::LayoutStride`
4. **`multi_slice_view`**: Multiple disconnected fluid regions — requires gather/scatter or temporary Views
5. **`optional_view`** and **`predicate_view`**: Conditional filtering for boundary/ghost regions

**Selector Mechanism:**

Selectors compose range-v3 views to target subsets of field data:
```cpp
// Domain interior selection
u | sel::D = initial_value;

// Dirichlet boundary selection
u | m.dirichlet(grid_bcs, object_bcs) = exact_solution;

// Fluid region (skipping solid cells)
u | m.fluid_all(object_bcs) += source_term;
```

**Key Range-v3 Patterns:**
```cpp
// Zip scalar components for element-wise operations
vs::zip(FWD(t).scalars()...)
vs::zip_with(f, FWD(t).scalars()...)

// View adaptors for non-contiguous access
class plane_view<I> : public rs::view_adaptor<plane_view<I>, Rng> { ... };

// Algorithms
rs::for_each, rs::copy, rs::fill, rs::equal, rs::minmax, rs::swap_ranges

// Views
vs::zip, vs::zip_with, vs::drop_exactly, vs::take_exactly, vs::stride,
vs::repeat_n, vs::common, vs::all, vs::transform
```

### 5.4 Stencils Subsystem

**Files:** `src/stencils/stencil.hpp/cpp`, `E2_1.cpp`, `E2_2.cpp`, `E4_2.cpp`, `E4u_1.cpp`, `E6u_1.cpp`, `E8u_1.cpp`, `polyE2_1.cpp`

**Naming Convention:** `E[accuracy-order][u]_[derivative-order]`

| Stencil | Accuracy | Derivative | Interior Width | Boundary Rows | Notes |
|---------|----------|-----------|----------------|---------------|-------|
| E2_1 | 2nd order | 1st | 3-point | 4 | Standard SBP |
| E2_2 | 2nd order | 2nd | 3-point | 2 | Laplacian |
| E4_2 | 4th order | 2nd | 5-point | 3 | High-order Laplacian |
| E4u_1 | 4th order | 1st | — | — | Upwind variant |
| E6u_1 | 6th order | 1st | — | — | Upwind variant |
| E8u_1 | 8th order | 1st | — | — | Upwind variant |
| polyE2_1 | 2nd order | 1st | — | — | Parameterized coefficients |

**Stencil Structure:**
```cpp
class stencil {
    std::vector<real> interior_coefficients;  // Interior stencil weights
    std::vector<real> left_boundary;          // Left boundary dense block
    std::vector<real> right_boundary;         // Right boundary dense block
    integer interior_size;                    // Stencil width
    integer boundary_rows;                    // Number of non-standard boundary rows
};
```

**Cut-Cell Modification:** Near embedded objects, stencil coefficients are modified based on the `psi` parameter. The `polyE2_1` stencil demonstrates parameterized coefficients where cut-cell geometry directly influences the finite difference weights.

**Range-v3 in Stencils:**
```cpp
// Extracting boundary stencil rows via chunking
lc | vs::chunk(tLeft) | vs::for_each(vs::drop(1))

// Computing cut-cell coefficients
c | vs::take_exactly(tObj)
c | vs::drop_exactly((rObj-1)*tObj) | vs::take_exactly(tObj) | vs::reverse
```

### 5.5 Operators Subsystem

**Files:** `src/operators/derivative.hpp/cpp`, `gradient.hpp/cpp`, `laplacian.hpp/cpp`, `divergence.hpp`, `directional.hpp/cpp`, `boundaries.hpp/cpp`, `discrete_operator.hpp`, `identity_stencil.hpp`, `eigenvalue_visitor.hpp/cpp`, `operator_visitor.hpp`

**Operator Hierarchy:**

```
discrete_operator (base concept)
├── derivative      — 1D finite difference along one direction
├── gradient        — [∂/∂x, ∂/∂y, ∂/∂z] composition of derivatives
├── laplacian       — ∂²/∂x² + ∂²/∂y² + ∂²/∂z² composition
├── divergence      — ∇·F composition (for conservation laws)
└── directional     — Complex geometry-dependent operator
```

**Operator Application Pipeline:**

1. Select grid line (1D slice through 3D domain)
2. Apply boundary conditions to line endpoints
3. Apply `inner_block` matrix: `[dense_left | circulant_interior | dense_right]`
4. Accumulate result into output field
5. Repeat for all lines in all active directions

**Derivative Construction:**
```cpp
// For each grid line in direction `dir`:
inner_block = {
    left_boundary:  dense matrix from stencil boundary coefficients,
    interior:       circulant matrix from stencil interior coefficients,
    right_boundary: dense matrix from stencil boundary coefficients
};

// Cut-cell lines get modified:
inner_block_cutcell = {
    left_boundary:  dense (modified by psi),
    interior:       circulant (standard),
    right_boundary: dense (modified by psi)
};
```

**Boundary Operator:**

The `boundaries` class handles the BC matrix `B` that maps boundary values into the operator:
```cpp
struct boundaries {
    matrix::csr B;      // Domain boundary operator
    matrix::csr Bfx;    // Object boundary (forward, x-dir)
    matrix::csr Brx;    // Object boundary (reverse, x-dir)
    // ... Bfy, Bry, Bfz, Brz for other directions
};
```

**Range-v3 in Operators:**
```cpp
// Enumerate shapes for cut-cell processing
for (auto&& [shape_row, obj] : vs::enumerate(shapes)) { ... }

// Transform solid points to direction-local indices
g.S(dir) | ranges::view::transform(m.ucf_ijk2dir(dir))
         | ranges::view::take(b_sz)
         | ranges::to<std::vector<int>>()

// Reduction over boundary conditions
rs::accumulate(obj_bcs, true, [](auto&& acc, auto&& cur) { ... })
```

### 5.6 Matrices Subsystem

**Files:** `src/matrices/csr.hpp/cpp`, `dense.hpp/cpp`, `circulant.hpp/cpp`, `inner_block.hpp/cpp`, `block.hpp`, `common.hpp`, `matrix_visitor.hpp`, `coefficient_visitor.hpp/cpp`, `unit_stride_visitor.hpp/cpp`

**Matrix Types:**

| Type | Storage | Purpose | Kokkos Equivalent |
|------|---------|---------|-------------------|
| `csr` | 3-array CSR (w, v, u) | Sparse boundary matrices | `Kokkos::CrsMatrix` or custom |
| `dense` | Row-major `vector<real>` | Boundary stencil blocks | `Kokkos::View<real**,LayoutRight>` |
| `circulant` | Shared stencil coefficients | Interior repeated stencil | Kernel with offset arithmetic |
| `inner_block` | `[dense \| circulant \| dense]` | Full 1D line operator | Composite kernel |
| `block` | `vector<inner_block>` | Multi-line operator | Multi-launch or fused kernel |

**Matrix-Vector Product Patterns:**

All matrices implement:
```cpp
template <typename Op = eq_t>
void operator()(std::span<const real> x, std::span<real> b, Op op = {}) const;
```

**CSR SpMV:**
```cpp
for (integer row = 0; row < rows(); row++)
    for (integer i = u[row]; i < u[row+1]; i++)
        b[row] += w[i] * x[v[i]];
```

**Dense MatVec (range-v3):**
```cpp
auto rng = vs::zip_with(
    [](auto&& a, auto&& b) { return rs::inner_product(a, b, 0.0); },
    vs::chunk(v, columns()),        // Split flat storage into rows
    vs::repeat_n(x, rows())         // Broadcast input vector
);
for (auto&& [y, z] : vs::zip(b, rng)) op(y, z);
```

**Circulant Stencil Application (range-v3):**
```cpp
auto rng = vs::zip_with(
    [](auto&& a, auto&& b) { return rs::inner_product(a, b, 0.0); },
    vs::repeat_n(v, rows()),        // Repeat stencil coefficients
    x | vs::sliding(size())         // Sliding window over input
);
for (auto&& [y, z] : vs::zip(b, rng)) op(y, z);
```

**Strided Application (for multi-D operators along non-X directions):**
```cpp
auto in = x | vs::stride(st);
auto out = b | vs::stride(st);
// Apply stencil to strided data
```

**Visitor Pattern:**

Two visitors extract information from matrix hierarchies:
- **`unit_stride_visitor`:** Maps matrix row/column indices to output DOF indices, handling cut-cell DOF removal
- **`coefficient_visitor`:** Extracts matrix coefficients into a dense global view using the index mapping

### 5.7 Systems Subsystem

**Files:** `src/systems/system.hpp/cpp`, `heat.hpp/cpp`, `scalar_wave.hpp/cpp`, `cc_elliptic.hpp/cpp`, `inviscid_vortex.hpp/cpp`, `hyperbolic_eigenvalues.hpp/cpp`, `empty_system.hpp/cpp`

**Type-Erased System Interface:**
```cpp
class system {
    std::variant<systems::empty, systems::scalar_wave,
                 systems::inviscid_vortex, systems::heat,
                 systems::hyperbolic_eigenvalues> v;

    // Key methods (dispatched via std::visit):
    field operator()(const step_controller&);           // Initial conditions
    void rhs(field_view f, real time, field_span rhs);  // Spatial RHS
    void update_boundary(field_span f, real time);      // BC enforcement
    real timestep_size(const field&, const step_controller&); // CFL
    system_stats stats(const field&, const field&, const step_controller&);
};
```

**Heat System RHS:**
```cpp
void heat::rhs(field_view f, real time, field_span rhs) const {
    auto&& u_rhs = rhs.scalars(scalars::u);
    auto&& u = f.scalars(scalars::u);

    u_rhs = lap(u, neumann_u);       // Discrete Laplacian
    u_rhs *= diffusivity;            // Scale by κ

    if (m_sol) {
        // MMS source term: S = ∂Q/∂t - κ∇²Q
        const auto src = (m.xyz | m_sol.ddt(time))
                       - (diffusivity * (m.xyz | m_sol.laplacian(time)));
        u_rhs | m.fluid_all(object_bcs) += src;
        u_rhs | m.dirichlet(grid_bcs, object_bcs) = 0;
    }
}
```

**Scalar Wave System RHS:**
```cpp
void scalar_wave::rhs(field_view f, real, field_span rhs) {
    auto&& u = f.scalars(scalars::u);
    auto&& u_rhs = rhs.scalars(scalars::u);

    du = grad(u);                    // Discrete gradient
    u_rhs = dot(grad_G, du);         // Dot with wave speed vector
}
```

**Statistics Computation (heavy range-v3):**
```cpp
auto [u_min, u_max] = minmax(u | m.fluid_all(object_bcs));
real err = max(abs(u - sol) | m.fluid_all(object_bcs));
// Uses rs::max_element, rs::distance, transform, accumulate
```

### 5.8 Temporal Integration

**Files:** `src/temporal/integrator.hpp/cpp`, `euler.hpp/cpp`, `rk4.hpp/cpp`, `step_controller.hpp/cpp`, `empty_integrator.hpp/cpp`

**RK4 Implementation:**
```cpp
constexpr std::array rki{0.0, 0.5, 0.5, 1.0};                    // Stage time offsets
constexpr std::array rkf{1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0};  // Stage weights

void rk4::operator()(system& sys, const field& u0, field_span u,
                     const step_controller& ctrl, real dt) {
    rk_rhs = 0; system_rhs = 0;
    const real time = ctrl;
    u = u0;

    for (int i = 0; i < 4; ++i) {
        if (i > 0) {
            u = u0 + dt * rki[i] * system_rhs;
            sys.update_boundary(u, time + dt * rki[i]);
        }
        system_rhs = sys.rhs(u, time + dt * rki[i]);
        rk_rhs += dt * rkf[i] * system_rhs;
    }
    u = u0 + rk_rhs;
    sys.update_boundary(u, time + dt);
}
```

**Step Controller:**
```cpp
class step_controller {
    bounded<int> step;     // Current iteration [0, max_step]
    bounded<real> time;    // Current time [0, max_time]
    real h_cfl;            // Hyperbolic CFL number
    real p_cfl;            // Parabolic CFL number
    real min_dt;           // Minimum allowed timestep
};
```

### 5.9 Simulation Orchestration

**Files:** `src/simulation/simulation_builder.hpp/cpp`, `simulation_cycle.hpp/cpp`, `src/app/shoccs.cpp`, `src/lib/run_from_sol.cpp`

**Control Flow:**
```
main() → parse CLI args → load Lua config
       → simulation_builder::from_lua(config)
           ├── system::from_lua(config)
           ├── integrator::from_lua(config)
           ├── step_controller::from_lua(config)
           └── field_io::from_lua(config)
       → simulation_cycle::run()
           └── while (step_controller) {
                   dt = system.timestep_size(u, step)
                   u_new = integrator(system, u, step, dt)
                   stats = system.stats(u_old, u_new, step)
                   io.write_if_needed(system, u_new, step, dt)
                   step.advance(dt)
               }
       → return final stats
```

**Two-Field Swap Pattern:** The cycle alternates between two field buffers to avoid allocation during time stepping.

### 5.10 I/O Subsystem

**Files:** `src/io/field_io.hpp/cpp`, `field_data.hpp/cpp`, `xdmf.hpp/cpp`, `logging.hpp/cpp`, `interval.hpp`

**Output Format:** XDMF metadata (XML) + binary data files. The XDMF file describes a time series of solution snapshots; each snapshot's field data is written as raw binary.

**Range-v3 in I/O:**
```cpp
// File name generation
vs::transform(names, [](auto&& n) { return n + ".bin"; })

// Parallel file iteration
vs::zip(file_handles, field_data) | for_each(write_binary)
```

**Interval System:** Output controlled by time interval or step interval, configured via Lua.

### 5.11 MMS (Verification)

**Files:** `src/mms/manufactured_solutions.hpp`, `mms.cpp`, `gauss.hpp`, `gauss1d/2d/3d.cpp`, `lua_mms.hpp/cpp`

**Manufactured Solution Interface:**
```cpp
class manufactured_solution {
    // Evaluate solution: Q(t, x)
    auto operator()(real time) const;
    // Time derivative: ∂Q/∂t
    auto ddt(real time) const;
    // Spatial gradient: ∂Q/∂x_i
    auto gradient(int dir, real time) const;
    // Laplacian: ∇²Q
    auto laplacian(real time) const;
};
```

**Gaussian Implementation:** Provides analytical derivatives for verification:
```
Q(t,x) = A · exp(-((x-cx)²/(2σx²) + (y-cy)²/(2σy²))) · cos(2πft)
```

All derivatives computed analytically — no numerical differentiation.

**Lua-defined MMS:** Allows arbitrary manufactured solutions defined in Lua with symbolic differentiation via sol2.

---

## 6. Complete Range-v3 Usage Catalog

### 6.1 View Types Used

| View | Files | Kokkos Replacement |
|------|-------|-------------------|
| `vs::zip` | fields, matrices, operators, I/O | Explicit multi-array loops or `Kokkos::parallel_for` with multiple View args |
| `vs::zip_with` | matrices (dense, circulant) | `Kokkos::parallel_for` with lambda combining arrays |
| `vs::transform` | fields, operators, I/O, statistics | `Kokkos::parallel_for` with output View |
| `vs::chunk(n)` | matrices (dense), visitors | Manual `i*n..(i+1)*n` indexing in kernel |
| `vs::sliding(n)` | matrices (circulant) | Manual `i..i+n` window in kernel |
| `vs::stride(n)` | matrices, operators | `Kokkos::subview` with LayoutStride or manual `i*stride` |
| `vs::repeat_n(v,n)` | matrices | Broadcast pattern in kernel |
| `vs::drop_exactly(n)` | fields, stencils | Offset in subview: `subview(v, make_pair(n, extent))` |
| `vs::take_exactly(n)` | fields, stencils | Size-limited subview: `subview(v, make_pair(0, n))` |
| `vs::common` | fields | Not needed (Kokkos views are always common) |
| `vs::all` | fields | Not needed (Kokkos views are already views) |
| `vs::enumerate` | operators, visitors | `Kokkos::parallel_for` index is implicit |
| `vs::reverse` | stencils | Reverse loop or reversed subview |
| `vs::for_each` | visitors (nested composition) | Nested loops |
| `vs::linear_distribute` | mesh | Manual: `min + i*(max-min)/(n-1)` |
| `vs::cartesian_product` | mesh | `MDRangePolicy<Rank<3>>` |

### 6.2 Algorithms Used

| Algorithm | Files | Kokkos Replacement |
|-----------|-------|-------------------|
| `rs::inner_product` | matrices (dense, circulant) | `KokkosBlas::dot()` or manual reduction |
| `rs::copy` | fields, matrices, mesh | `Kokkos::deep_copy` |
| `rs::fill` | fields | `Kokkos::deep_copy(view, value)` |
| `rs::for_each` | fields, operators | `Kokkos::parallel_for` |
| `rs::sort` | matrices (CSR builder) | `Kokkos::sort` or host-side `std::sort` |
| `rs::equal` | tests | `Kokkos::parallel_reduce` with comparison |
| `rs::minmax` | statistics | `Kokkos::parallel_reduce` with MinMax reducer |
| `rs::min` | step controller | `Kokkos::parallel_reduce` with Min reducer |
| `rs::max_element` | statistics | `Kokkos::parallel_reduce` with MaxLoc reducer |
| `rs::accumulate` | operators | `Kokkos::parallel_reduce` with Sum reducer |
| `rs::count_if` | mesh | `Kokkos::parallel_reduce` with count |
| `rs::distance` | statistics | Index arithmetic |
| `rs::to<Container>` | operators, mesh | Manual loop or host-side conversion |
| `rs::shuffle` | tests | `std::shuffle` (host only) |

### 6.3 Custom Adaptors

| Adaptor | Location | Complexity | Migration Approach |
|---------|----------|------------|-------------------|
| `plane_view<0>` (X) | fields | Medium | `Kokkos::subview` (contiguous) |
| `plane_view<1>` (Y) | fields | **High** | Custom kernel with strided access |
| `plane_view<2>` (Z) | fields | Medium | `Kokkos::subview` with LayoutStride |
| `multi_slice_view` | fields | **High** | Gather into temporary View, or index list |
| `optional_view` | fields | Medium | Conditional mask in kernel |
| `predicate_view` | fields | Medium | Filter with index array |
| `semiregular_box_t` | fields | Low | Not needed in Kokkos |

### 6.4 Coroutine Generators

| Generator | Location | Replacement |
|-----------|----------|-------------|
| `index_view(int3)` | indexing | `Kokkos::MDRangePolicy<Rank<3>>` |
| Line iteration | mesh | `Kokkos::RangePolicy` per direction |

---

## 7. Data Flow: End-to-End

### 7.1 Initialization

```
Lua Config
  ├── mesh::from_lua → Cartesian grid + cut-cell geometry
  │     ├── Compute coordinates (vs::linear_distribute)
  │     ├── Ray-cast objects → Rx, Ry, Rz intersection data
  │     └── Build fluid slices, solid masks, selection views
  ├── stencil::from_lua → Finite difference coefficients
  ├── bcs::from_lua → Boundary condition types per face
  ├── system::from_lua → PDE system with operators
  │     ├── Build derivative operators (derivative → inner_block → block)
  │     ├── Compose into gradient/laplacian/divergence
  │     └── Initialize manufactured solution
  ├── integrator::from_lua → RK4 or Euler
  ├── step_controller::from_lua → CFL numbers, termination criteria
  └── field_io::from_lua → Output intervals, file paths
```

### 7.2 Per-Timestep Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  TIMESTEP n                                                      │
│                                                                  │
│  1. CFL check:  dt = system.timestep_size(u, step_ctrl)         │
│     └── rs::min(mesh.h()) → minimum grid spacing                 │
│     └── dt = CFL * h² / (4κ)  or  CFL * h / c                  │
│                                                                  │
│  2. RK4 stages (4 evaluations):                                  │
│     For stage i = 0..3:                                          │
│       ├── u_stage = u0 + dt * rk_coeff[i] * k_prev              │
│       ├── update_boundary(u_stage, t + dt*rk_time[i])            │
│       │   └── u | m.dirichlet(...) = exact_solution              │
│       │   └── neumann_data | m.neumann<dir>(...) = gradient      │
│       ├── k_i = system.rhs(u_stage, t + dt*rk_time[i])          │
│       │   └── For heat: k = κ·lap(u) + source                   │
│       │       └── lap(u): iterate all grid lines                 │
│       │           └── For each line: inner_block(x, b)           │
│       │               ├── dense_left: small MatVec               │
│       │               ├── circulant: stencil application         │
│       │               │   └── vs::sliding + rs::inner_product    │
│       │               └── dense_right: small MatVec              │
│       │   └── For wave: k = dot(grad_G, grad(u))                │
│       │       └── grad(u): derivative in each direction          │
│       └── rk_rhs += dt * rk_weight[i] * k_i                     │
│                                                                  │
│  3. Solution update:  u_new = u0 + rk_rhs                       │
│     └── update_boundary(u_new, t + dt)                           │
│                                                                  │
│  4. Statistics:  stats = system.stats(u_old, u_new, step)        │
│     └── error = max(|u - u_exact|) over fluid domain             │
│     └── rs::max_element, rs::minmax, rs::accumulate              │
│                                                                  │
│  5. I/O:  if interval triggered → write XDMF + binary           │
│                                                                  │
│  6. Advance:  step_ctrl.advance(dt)                              │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 Memory Access Pattern Summary

| Operation | Access Pattern | Parallelism Model |
|-----------|---------------|-------------------|
| Circulant stencil | Sliding window, sequential | `parallel_for` over rows |
| Dense boundary MatVec | Row-major sequential | `parallel_for` over rows |
| CSR SpMV | Row-sequential, column-gather | `parallel_for` over rows |
| Multi-D operator | Strided 1D slices through 3D array | `parallel_for` over lines |
| Field arithmetic | Element-wise contiguous | `parallel_for` over elements |
| Statistics reduction | Sequential scan with min/max | `parallel_reduce` |
| BC enforcement | Scattered writes to boundary indices | `parallel_for` over BC indices |

---

## 8. Kokkos Migration Strategy

### 8.1 Data Container Migration

**Tier 0 — Foundation:**
```cpp
// BEFORE
std::vector<real> data;
std::span<real> view;

// AFTER
Kokkos::View<real*> data("data", n);
Kokkos::View<real*, Kokkos::HostSpace> host_data("host_data", n);
// Or dual views for host/device:
Kokkos::DualView<real*> data("data", n);
```

### 8.2 Iteration Pattern Migration

**Simple loops → parallel_for:**
```cpp
// BEFORE
rs::for_each(vs::zip(a, b), [](auto [x, y]) { x += y; });

// AFTER
Kokkos::parallel_for(n, KOKKOS_LAMBDA(int i) {
    a(i) += b(i);
});
```

**Multi-dimensional iteration → MDRangePolicy:**
```cpp
// BEFORE (coroutine generator)
for (auto ijk : index_view(extents)) { ... }

// AFTER
Kokkos::parallel_for(
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nx,ny,nz}),
    KOKKOS_LAMBDA(int i, int j, int k) { ... }
);
```

**Stencil application → parallel_for with local reduction:**
```cpp
// BEFORE
auto rng = vs::zip_with(inner_product,
    vs::repeat_n(stencil, rows),
    input | vs::sliding(width));

// AFTER
Kokkos::parallel_for(rows, KOKKOS_LAMBDA(int i) {
    real sum = 0.0;
    for (int j = 0; j < width; j++)
        sum += stencil(j) * input(i + j);
    output(i) = sum;
});
```

**Strided access → subview or LayoutStride:**
```cpp
// BEFORE
auto in = x | vs::stride(st);

// AFTER
auto in = Kokkos::subview(x, Kokkos::make_pair(0, n*st, st));
// Or manual indexing:
Kokkos::parallel_for(n, KOKKOS_LAMBDA(int i) {
    output(i) = stencil_apply(x, i * stride, width);
});
```

### 8.3 Reduction Migration

```cpp
// BEFORE
auto [mn, mx] = rs::minmax(u | m.fluid_all(bcs));

// AFTER
real mn, mx;
Kokkos::parallel_reduce(n,
    KOKKOS_LAMBDA(int i, real& lmin, real& lmax) {
        if (is_fluid(i)) {
            lmin = Kokkos::min(lmin, u(i));
            lmax = Kokkos::max(lmax, u(i));
        }
    },
    Kokkos::Min<real>(mn), Kokkos::Max<real>(mx)
);
```

### 8.4 Field/Selector Migration

The selector mechanism (`u | m.dirichlet(...)`) must be replaced with either:

**Option A: Index list approach**
```cpp
// Pre-compute boundary index lists
Kokkos::View<int*> dirichlet_indices("dirichlet_idx", n_dirichlet);
// Fill once during initialization

// Apply BC
Kokkos::parallel_for(n_dirichlet, KOKKOS_LAMBDA(int ii) {
    int i = dirichlet_indices(ii);
    u(i) = exact_solution(i);
});
```

**Option B: Mask approach**
```cpp
// Pre-compute boolean mask
Kokkos::View<bool*> is_dirichlet("is_dirichlet", n_total);

// Apply BC
Kokkos::parallel_for(n_total, KOKKOS_LAMBDA(int i) {
    if (is_dirichlet(i))
        u(i) = exact_solution(i);
});
```

Option A is preferred for sparse selections (fewer divergent threads on GPU).

### 8.5 Matrix Format Migration

| Current | Kokkos Replacement |
|---------|-------------------|
| `csr` (3-array) | `KokkosSparse::CrsMatrix<real>` |
| `dense` (row-major vector) | `Kokkos::View<real**, LayoutRight>` |
| `circulant` (stencil span) | Custom kernel with `Kokkos::View<real*>` coefficients |
| `inner_block` | Fused kernel or 3 sequential launches |
| `block` | `Kokkos::parallel_for` over lines with team policy |

### 8.6 Visitor Pattern Adaptation

The visitor pattern for matrix assembly can be preserved on the host side (assembly is typically done once at initialization). The assembled matrices are then copied to device memory:

```cpp
// Assembly remains on host
auto block_h = build_operator_host(mesh, stencil, bcs);

// Copy to device
auto block_d = Kokkos::create_mirror_view_and_copy(
    Kokkos::DefaultExecutionSpace(), block_h);
```

---

## 9. Migration Roadmap

### Phase 1: Foundation (Week 1-2)
- Add Kokkos dependency to CMake build system
- Replace `types.hpp` aliases with Kokkos-compatible types
- Create `Kokkos::View` wrappers for core data containers
- Migrate `index_extents` and linear indexing to Kokkos-compatible functions
- Replace coroutine generators with MDRangePolicy iteration

### Phase 2: Fields & Selectors (Week 2-3)
- Migrate `scalar_real` storage from `std::vector<real>` to `Kokkos::View<real*>`
- Replace custom view adaptors (plane_view, multi_slice_view) with subviews and index lists
- Implement selector mechanism using pre-computed index arrays
- Migrate field arithmetic to `Kokkos::parallel_for`
- Port tuple operations to work with Kokkos views

### Phase 3: Matrices (Week 3-4)
- Migrate CSR to `KokkosSparse::CrsMatrix` or custom Kokkos CSR
- Migrate dense matrix to `Kokkos::View<real**, LayoutRight>`
- Rewrite circulant stencil application as Kokkos kernel
- Adapt inner_block and block to launch Kokkos kernels
- Keep visitor-based assembly on host, copy to device

### Phase 4: Operators (Week 4-5)
- Port derivative operator to use Kokkos matrix kernels
- Migrate gradient/laplacian/divergence composition
- Port boundary operator with Kokkos CSR SpMV
- Adapt directional operator for complex geometry
- Validate operator accuracy against reference (MMS convergence tests)

### Phase 5: Systems & Temporal (Week 5-6)
- Port RHS evaluation for heat and scalar_wave systems
- Adapt RK4/Euler time integration loop
- Migrate statistics computation to Kokkos reductions
- Port BC enforcement using index list approach
- Maintain Lua configuration interface unchanged

### Phase 6: I/O, MMS & Integration Testing (Week 6-8)
- Adapt I/O to read from device memory (deep_copy to host for output)
- Ensure MMS verification produces identical convergence rates
- Run full simulation regression tests
- Performance benchmarking: Kokkos (CPU threads, GPU) vs. original range-v3
- Documentation and cleanup

### Estimated Total Effort: 6-8 engineer-weeks

### Risk Areas

| Risk | Severity | Mitigation |
|------|----------|------------|
| Y-plane non-contiguous access | High | Custom kernel with explicit strided indexing |
| multi_slice_view replacement | High | Gather into temporary contiguous buffer |
| Lazy evaluation loss | Medium | Profile eager execution overhead; batch operations |
| Host/device data movement | Medium | Use DualView or Unified Memory initially |
| Template metaprogramming incompatibility | Medium | Simplify template patterns for KOKKOS_LAMBDA capture |
| Assembly on device vs host | Low | Keep assembly on host, copy once |

---

*This specification was generated by deep recursive analysis of all 189 source files across 16 subsystem directories in the SHOCCS codebase. Each subsystem was analyzed by a dedicated agent reading every header, implementation, and test file.*
