# Group Velocity Analysis Reference

## 1. Module Overview

**Module:** `stencil_gen.group_velocity`

This module provides tools for computing modified wavenumber, phase velocity, and group velocity from finite difference stencil coefficients. It supports interior stencils, boundary closures, cut-cell stencils with non-uniform spacing, 2D tensor-product operators, varying-coefficient problems, and GKS-type stability diagnostics.

### Mathematical Foundation

For the scalar advection equation

    u_t + u_x = 0

semi-discretized on a uniform grid as `du/dt = -D*u`, the stencil `D` acting on a Fourier mode `exp(i*j*xi)` produces a **modified wavenumber** `kappa*(xi)`:

    kappa*(xi) = sum_j  w_j  exp(i * (j - i_eval) * xi)

where `w_j` are the stencil weights and `i_eval` is the evaluation point index.

The **dispersion relation** is `omega = Im(kappa*(xi))`. From this:

- **Phase velocity:** `c(xi) = Im(kappa*(xi)) / xi` -- the speed at which wave crests of wavenumber `xi` propagate.
- **Group velocity:** `C(xi) = d(Im(kappa*(xi))) / d(xi)` -- the speed at which wave energy (packets) propagate.

For the exact PDE, `kappa*(xi) = i*xi`, so `c(xi) = C(xi) = 1` everywhere.

### Sign Conventions

- The PDE is `u_t + u_x = 0`, so the physical propagation is in the **positive x-direction** with unit speed.
- `C(xi) > 0` means the stencil propagates energy rightward (same direction as the PDE).
- `C(xi) < 0` means the stencil produces **spurious backwards-propagating** energy at wavenumber `xi`.
- The **cutoff wavenumber** is the first `xi` beyond which `C` stays permanently non-positive. High-frequency modes above the cutoff carry energy in the wrong direction.
- For an interior scheme of order `2p` (half-bandwidth `p`), group velocity error grows as `O(xi^(2p))` for small `xi`. The error magnitude is amplified by a factor of `(2p+1)` relative to the phase velocity error due to differentiation of the dispersion relation (see Key Mathematical Notes below).

---

## 2. Public API

### 2.1 Core Computations

These are the low-level building blocks. They operate on raw stencil weights and wavenumber arrays.

#### `modified_wavenumber(weights, i_eval, node_indices, xi_array)`

Compute `kappa*(xi) = sum_j w_j exp(i*(j - i_eval)*xi)` for integer grid indices.

| Parameter | Type | Description |
|---|---|---|
| `weights` | array-like | Stencil coefficients `w_j` |
| `i_eval` | `int` | Grid index where derivative is evaluated |
| `node_indices` | array-like of int | Grid indices used by the stencil |
| `xi_array` | `np.ndarray` | Wavenumber values in `[0, pi]` |

**Returns:** `np.ndarray` (complex) -- modified wavenumber `kappa*(xi)`.

---

#### `modified_wavenumber_nonuniform(weights, offsets, xi_array)`

Generalization for real-valued (non-integer) offsets, as arise in cut-cell grids.

| Parameter | Type | Description |
|---|---|---|
| `weights` | array-like | Stencil coefficients |
| `offsets` | array-like of float | Normalized distances `(x_j - x_i)/h` from evaluation point (may be fractional) |
| `xi_array` | `np.ndarray` | Wavenumber values in `[0, pi]` |

**Returns:** `np.ndarray` (complex) -- `kappa*(xi) = sum_j w_j exp(i * offset_j * xi)`.

---

#### `group_velocity_exact(weights, i_eval, node_indices, xi_array)`

Compute group velocity analytically (no numerical differentiation):

    C(xi) = Re(sum_j w_j * (j - i_eval) * exp(i*(j - i_eval)*xi))

| Parameter | Type | Description |
|---|---|---|
| `weights` | array-like | Stencil coefficients |
| `i_eval` | `int` | Evaluation point grid index |
| `node_indices` | array-like of int | Grid indices used |
| `xi_array` | `np.ndarray` | Wavenumber values |

**Returns:** `np.ndarray` (real) -- group velocity `C(xi)`.

---

#### `group_velocity_exact_nonuniform(weights, offsets, xi_array)`

Analytical group velocity for non-uniform offsets:

    C(xi) = Re(sum_j w_j * offset_j * exp(i * offset_j * xi))

Parameters and return are analogous to `modified_wavenumber_nonuniform`.

---

#### `group_velocity_numerical(kappa_star, xi_array)`

Compute `C(xi) = d(Im(kappa*))/d(xi)` via `np.gradient` (numerical differentiation). Less accurate than `group_velocity_exact` but works when you only have a precomputed `kappa*` array.

| Parameter | Type | Description |
|---|---|---|
| `kappa_star` | `np.ndarray` (complex) | Modified wavenumber array |
| `xi_array` | `np.ndarray` | Wavenumber values |

**Returns:** `np.ndarray` (real) -- group velocity.

---

#### `phase_velocity(kappa_star, xi_array)`

Compute phase velocity `c(xi) = Im(kappa*(xi)) / xi`. At `xi = 0` the singularity is handled by using the value at the first non-zero `xi` point.

**Returns:** `np.ndarray` (real) -- phase velocity.

---

#### `group_velocity_error(C, C_exact=1.0)`

Compute relative group velocity error: `(C - C_exact) / C_exact`. Default `C_exact = 1.0` matches the advection equation `u_t + u_x = 0`.

**Returns:** `np.ndarray` -- relative error.

---

### 2.2 Interior Analysis

#### `interior_group_velocity(p, nu, xi_array)`

Convenience function: derives the `2p`-order explicit interior stencil for derivative order `nu`, then returns a full `GroupVelocityProfile`.

| Parameter | Type | Description |
|---|---|---|
| `p` | `int` | Interior half-bandwidth (E2: `p=1`, E4: `p=2`, E6: `p=3`, E8: `p=4`) |
| `nu` | `int` | Derivative order (1 or 2) |
| `xi_array` | `np.ndarray` | Wavenumber values in `[0, pi]` |

**Returns:** `GroupVelocityProfile`

Internally calls `stencil_gen.interior.derive_interior(0, p, nu)` and `full_gamma_array` to get stencil weights, then delegates to `_build_profile`.

---

### 2.3 Boundary Analysis

#### `boundary_group_velocity(p, q, nextra, nu, sigma, kernel, xi_array)`

Compute group velocity profiles for **all boundary rows** using RBF/tension boundary stencil weights.

| Parameter | Type | Description |
|---|---|---|
| `p` | `int` | Interior half-bandwidth |
| `q` | `int` | Polynomial degree for boundary augmentation |
| `nextra` | `int` | Extra boundary rows/columns |
| `nu` | `int` | Derivative order |
| `sigma` | `float` | RBF shape / tension parameter |
| `kernel` | `str` | RBF kernel: `"tension"`, `"gaussian"`, or `"multiquadric"` |
| `xi_array` | `np.ndarray` | Wavenumber values |

**Returns:** `dict[int, GroupVelocityProfile]` -- keyed by boundary row index `0` to `r-1`.

Internally uses `stencil_gen.phs.uniform_boundary_weights_rbf` and `stencil_gen.temo.compute_dimensions` to determine stencil layout.

---

#### `boundary_group_velocity_classical(boundary_rows, alpha_values, order, xi_array)`

Compute group velocity profiles for **classical (non-RBF) boundary rows** from the symbolic `BoundaryRow` objects produced by `derive_boundary` or `solve_conservation`.

| Parameter | Type | Description |
|---|---|---|
| `boundary_rows` | `list[BoundaryRow]` | Symbolic boundary rows with alpha parameters |
| `alpha_values` | `dict` | Mapping from alpha symbols to numeric values (e.g., `{alpha_0: -0.77}`) |
| `order` | `int` | Polynomial accuracy order of the boundary scheme |
| `xi_array` | `np.ndarray` | Wavenumber values |

**Returns:** `dict[int, GroupVelocityProfile]` -- keyed by boundary row index.

This function evaluates symbolic coefficients via `xreplace` to obtain numeric weights.

---

### 2.4 Cut-Cell Analysis

#### `cut_cell_group_velocity(cut_cell_result, psi_sym, psi_val, alpha_values, xi_array, order=None)`

Compute group velocity profiles for all rows of a **cut-cell stencil** at a specific `psi` value.

| Parameter | Type | Description |
|---|---|---|
| `cut_cell_result` | `CutCellResult` | Symbolic cut-cell stencil from `derive_cut_cell_mathematica` or `derive_cut_cell_scheme` |
| `psi_sym` | `Symbol` | SymPy symbol for psi used in the result |
| `psi_val` | `float` | Numeric psi value in `[0, 1]` |
| `alpha_values` | `dict` | Alpha symbol to value mapping |
| `xi_array` | `np.ndarray` | Wavenumber values |
| `order` | `int`, optional | Polynomial accuracy order; defaults to `dims.r - 1` |

**Returns:** `dict[int, GroupVelocityProfile]` -- keyed by row index `0` to `R-1`.

The non-uniform offsets are computed from the cut-cell geometry: the wall is at offset `-(psi_val + i)` from row `i`, and the regular grid points are at offsets `j - i`.

---

#### `psi_sweep_group_velocity(scheme_params, psi_values, alpha_values, xi_array, order=None)`

Sweep psi across the cut-cell parameter range `[0, 1]`, deriving the stencil once and evaluating group velocity at each psi.

| Parameter | Type | Description |
|---|---|---|
| `scheme_params` | `SchemeParams` | Scheme parameters (`p`, `q`, `s`, `nextra`, `nu`, `zeros`) |
| `psi_values` | `np.ndarray` | Array of psi values to sweep |
| `alpha_values` | `dict` | Alpha symbol to value mapping (empty dict uses zeros) |
| `xi_array` | `np.ndarray` | Wavenumber values |
| `order` | `int`, optional | Polynomial accuracy order |

**Returns:** `PsiSweepResult`

This function also computes the interior group velocity and checks for **sign reversals** -- wavenumbers where a boundary row has `C > 0` while the interior has `C < 0`, indicating parasitic energy propagating backwards through the boundary closure.

Stencil derivation uses the singularity-free Mathematica workflow (`derive_cut_cell_mathematica`) when `scheme_params.zeros` is non-empty, otherwise the standard pipeline (`derive_cut_cell_scheme`).

---

### 2.5 2D Analysis

#### `group_velocity_2d(kappa_x_star, kappa_y_star, xi_array, eta_array, a=1.0, b=1.0)`

Compute 2D group velocity for **tensor-product stencils** where the dispersion relation factors as:

    omega = a * kappa_x*(xi) + b * kappa_y*(eta)

The group velocity vector is `(C_x, C_y)` where each component depends only on its respective wavenumber.

| Parameter | Type | Description |
|---|---|---|
| `kappa_x_star` | `np.ndarray` (complex) | Modified wavenumber for x-direction, shape `(N_xi,)` |
| `kappa_y_star` | `np.ndarray` (complex) | Modified wavenumber for y-direction, shape `(N_eta,)` |
| `xi_array` | `np.ndarray` | x-direction wavenumber array |
| `eta_array` | `np.ndarray` | y-direction wavenumber array |
| `a` | `float` | Wave speed in x-direction (default 1.0) |
| `b` | `float` | Wave speed in y-direction (default 1.0) |

**Returns:** `GroupVelocity2DResult`

All 2D arrays in the result have shape `(N_xi, N_eta)` using `indexing='ij'`.

---

#### `anisotropy_profile(p, nu, theta_array, xi_mag)`

Compute group speed and angle error vs propagation angle for an interior scheme.

For the advection equation `u_t + cos(theta)*u_x + sin(theta)*u_y = 0`, this evaluates the numerical group velocity at wavenumber magnitude `xi_mag` for each propagation angle `theta`.

| Parameter | Type | Description |
|---|---|---|
| `p` | `int` | Interior half-bandwidth |
| `nu` | `int` | Derivative order |
| `theta_array` | `np.ndarray` | Propagation angles in radians |
| `xi_mag` | `float` | Wavenumber magnitude in `[0, pi]` |

**Returns:** `AnisotropyResult`

The exact group velocity is `(cos(theta), sin(theta))` with unit speed. Grid anisotropy causes deviations -- for example, Trefethen (1982) shows that E2 has higher group speed at diagonal propagation (`theta = pi/4`) than axis-aligned propagation (`theta = 0`).

---

#### `boundary_group_velocity_2d(boundary_rows_x, interior_y, theta_array, xi_mag)`

Compute 2D group velocity at a boundary where x-direction uses boundary stencils and y-direction uses interior stencils.

| Parameter | Type | Description |
|---|---|---|
| `boundary_rows_x` | `dict[int, GroupVelocityProfile]` | Boundary profiles for x-direction |
| `interior_y` | `GroupVelocityProfile` | Interior profile for y-direction |
| `theta_array` | `np.ndarray` | Propagation angles |
| `xi_mag` | `float` | Wavenumber magnitude |

**Returns:** `dict[int, AnisotropyResult]` -- keyed by boundary row index.

This reveals whether the boundary distorts the group velocity angle, bending waves toward or away from the boundary.

---

### 2.6 Varying-Coefficient Analysis

#### `local_group_velocity(weights_func, x, xi_array)`

Compute **spatially varying** group velocity for problems like `u_t + a(x)*u_x = 0` where stencil coefficients depend on position.

| Parameter | Type | Description |
|---|---|---|
| `weights_func` | callable | `weights_func(x_val) -> (weights, offsets)` returning stencil weights and offsets at position `x_val` |
| `x` | `np.ndarray` | Grid point coordinates, shape `(N_x,)` |
| `xi_array` | `np.ndarray` | Wavenumber values, shape `(N_xi,)` |

**Returns:** `np.ndarray`, shape `(N_x, N_xi)` -- local group velocity `C[i_x, i_xi]`.

---

#### `ray_trace_group_velocity(C_field, x_grid, xi_array, xi_0, x_0, t_final, dt)`

Trace a ray through a spatially varying group velocity field by integrating the ray equations:

    dx/dt  =  C(x, xi)      (group velocity)
    dxi/dt = -dC/dx          (refraction)

using classical RK4 with bilinear interpolation of the `C_field`.

| Parameter | Type | Description |
|---|---|---|
| `C_field` | `np.ndarray` | Group velocity field from `local_group_velocity`, shape `(N_x, N_xi)` |
| `x_grid` | `np.ndarray` | Grid coordinates, shape `(N_x,)` |
| `xi_array` | `np.ndarray` | Wavenumber values, shape `(N_xi,)` |
| `xi_0` | `float` | Initial wavenumber |
| `x_0` | `float` | Initial position |
| `t_final` | `float` | Final integration time |
| `dt` | `float` | Time step |

**Returns:** `RayTraceResult`

Uses `scipy.interpolate.RegularGridInterpolator` for bilinear interpolation and `np.gradient` along axis 0 for `dC/dx`. The refraction equation (`dxi/dt = -dC/dx`) is from Trefethen (1982), Eq. 4.9b.

---

### 2.7 GKS Stability Diagnostic

#### `gks_group_velocity_check(D, xi_array, neutral_tol=0.1, localization_tol=0.3)`

Identify boundary-localized eigenmodes whose group velocity indicates **GKS-type instability**.

For the semi-discrete problem `du/dt = -D*u` with Dirichlet inflow BC (row/column 0 removed), this function:

1. Computes eigenvalues and eigenvectors of `-D_bc`.
2. Identifies **nearly-neutral** modes (small `|Re(lambda)|`).
3. Checks for **boundary localization** (energy concentrated in the first or last quarter of the domain).
4. Estimates the dominant wavenumber via zero-padded FFT of the boundary portion.
5. Evaluates the interior group velocity at that wavenumber.
6. Flags the mode as **outgoing** if the group velocity directs energy from the boundary into the domain -- the hallmark of GKS instability.

| Parameter | Type | Description |
|---|---|---|
| `D` | `np.ndarray` | Full `N x N` differentiation matrix |
| `xi_array` | `np.ndarray` | Wavenumber array for group velocity evaluation |
| `neutral_tol` | `float` | Fraction of `max|Re(lambda)|` below which a mode is nearly-neutral (default 0.1) |
| `localization_tol` | `float` | Minimum fraction of eigenvector energy in the boundary region to classify as boundary-localized (default 0.3) |

**Returns:** `list[GKSModeInfo]` -- one entry per identified mode.

The left boundary is "outgoing" when `C > 0` (rightward into domain); the right boundary is "outgoing" when `C < 0` (leftward into domain). Only positive-imaginary members of conjugate pairs are kept.

**Heuristic vs. rigorous test:** `gks_group_velocity_check` is a *heuristic* diagnostic (necessary but not sufficient for instability). For the rigorous GKS determinant condition, use `kreiss_stability_check` from `stencil_gen.gks_kreiss`, which implements Trefethen 1983 pp. 206-207: it sweeps the right half-plane for `s` where `sigma_min(M(s)) < tol`, refines witnesses via Nelder-Mead, and classifies imaginary-axis modes as incoming or outgoing. See `docs/brady2d_stability_reference.md` for the full API.

---

#### `local_group_velocity_2d_varying(interior_stencil_x, interior_stencil_y, c_x_field, c_y_field, xi_array)`

Compute **per-point local group velocity error** across a 2D varying-coefficient field. At each grid point `(i, j)`, the local dispersion error is `c_*[i,j] * gv_error(xi)` — the factor of `c_*` scales the baseline interior error by the local wave speed. This is the first-order WKB approximation for smooth coefficient fields.

Used by Layer 4 (L4) of the Brady-Livescu 2D stability pipeline. See `docs/brady2d_stability_reference.md`.

| Parameter | Type | Description |
|---|---|---|
| `interior_stencil_x` | tuple | `(weights, offsets)` for x-direction interior stencil |
| `interior_stencil_y` | tuple | `(weights, offsets)` for y-direction interior stencil |
| `c_x_field` | `np.ndarray` | x-component of varying coefficient field, shape `(Ny, Nx)` |
| `c_y_field` | `np.ndarray` | y-component of varying coefficient field, shape `(Ny, Nx)` |
| `xi_array` | `np.ndarray` | Wavenumber array |

**Returns:** `dict` with keys `C_x_field`, `C_y_field`, `gv_error_x_field`, `gv_error_y_field` (all shape `(Ny, Nx, N_xi)`) and scalar reduction `max_local_gv_error_2d(result) -> float`.

---

#### `anisotropy_over_coefficient_field(scheme, c_x_field, c_y_field, theta_array, xi_mag)`

Evaluate the directional alignment between the grid anisotropy profile and the local propagation direction across a 2D coefficient field. At each grid point, the radial propagation direction is `(c_x[i,j], c_y[i,j])/|c|` — the anisotropy error is projected onto this direction.

Used by Layer 5 (L5) of the Brady-Livescu 2D stability pipeline. See `docs/brady2d_stability_reference.md`.

| Parameter | Type | Description |
|---|---|---|
| `scheme` | `str` | Scheme name (e.g. "E2", "E4") for looking up order `p` |
| `c_x_field` | `np.ndarray` | x-component of varying coefficient field |
| `c_y_field` | `np.ndarray` | y-component of varying coefficient field |
| `theta_array` | `np.ndarray` | Propagation angles |
| `xi_mag` | `float` | Wavenumber magnitude |

**Returns:** `dict` with `max_aligned_error`, `worst_point`, `worst_theta`.

---

## 3. Dataclasses

### `GroupVelocityProfile`

Complete group velocity analysis for a single stencil row.

| Field | Type | Description |
|---|---|---|
| `xi` | `np.ndarray` | Wavenumber array |
| `kappa_star` | `np.ndarray` (complex) | Modified wavenumber `kappa*(xi)` |
| `phase_velocity` | `np.ndarray` | Phase velocity `c(xi) = Im(kappa*)/xi` |
| `group_velocity` | `np.ndarray` | Group velocity `C(xi) = d(Im(kappa*))/d(xi)` |
| `gv_error` | `np.ndarray` | Relative GV error `(C - 1) / 1` |
| `order` | `int` | Polynomial accuracy order of the stencil |
| `cutoff_xi` | `float` | First `xi` beyond which `C` stays permanently non-positive |

The cutoff is determined by scanning from `xi = pi` backward to find the last `xi` where `C > 0`, then taking the next grid point. This handles non-monotonic boundary stencils where `C` may dip below zero briefly then recover.

---

### `PsiSweepResult`

Results from sweeping psi across the cut-cell parameter range.

| Field | Type | Description |
|---|---|---|
| `psi_values` | `np.ndarray` | Array of psi values swept |
| `profiles` | `dict[float, dict[int, GroupVelocityProfile]]` | Nested dict: `{psi_val: {row_index: profile}}` |
| `worst_row` | `int` | Row index with the largest GV error across all psi values |
| `worst_psi` | `float` | Psi value with the largest GV error |
| `min_C` | `float` | Most negative group velocity across all psi/row combinations |
| `has_sign_reversal` | `bool` | True if any boundary row has `C > 0` where the interior has `C < 0` |

---

### `GroupVelocity2DResult`

2D group velocity analysis for tensor-product stencils.

| Field | Type | Description |
|---|---|---|
| `xi` | `np.ndarray` | 1D wavenumber array for x-direction |
| `eta` | `np.ndarray` | 1D wavenumber array for y-direction |
| `C_x` | `np.ndarray` | 2D group velocity x-component, shape `(N_xi, N_eta)` |
| `C_y` | `np.ndarray` | 2D group velocity y-component, shape `(N_xi, N_eta)` |
| `speed` | `np.ndarray` | 2D group speed `|C|` |
| `angle` | `np.ndarray` | 2D group propagation angle `atan2(C_y, C_x)` |
| `angle_error` | `np.ndarray` | Angle deviation from wave normal direction |

---

### `AnisotropyResult`

Anisotropy profile for a given scheme at fixed wavenumber magnitude.

| Field | Type | Description |
|---|---|---|
| `theta` | `np.ndarray` | Propagation angle array |
| `C_x` | `np.ndarray` | Group velocity x-component |
| `C_y` | `np.ndarray` | Group velocity y-component |
| `speed` | `np.ndarray` | Group speed `|C|` (equals speed ratio since exact speed is 1) |
| `angle` | `np.ndarray` | Group propagation angle `atan2(C_y, C_x)` |
| `angle_error` | `np.ndarray` | Angle deviation from propagation direction `theta` |

---

### `GKSModeInfo`

Diagnostic information for a boundary-localized eigenmode.

| Field | Type | Description |
|---|---|---|
| `eigenvalue` | `complex` | Eigenvalue of `-D_bc` |
| `boundary_wavenumber` | `float` | Dominant `xi` from FFT of boundary eigenvector portion |
| `group_velocity` | `float` | Interior `C(xi)` at `boundary_wavenumber` |
| `is_outgoing` | `bool` | True if the mode radiates energy from the boundary into the domain |

---

### `RayTraceResult`

Trajectory of a ray traced through a group velocity field.

| Field | Type | Description |
|---|---|---|
| `t` | `np.ndarray` | Time array, shape `(N_t,)` |
| `x` | `np.ndarray` | Position trajectory, shape `(N_t,)` |
| `xi` | `np.ndarray` | Wavenumber trajectory, shape `(N_t,)` |

---

## 4. Usage Examples

### 4.1 Interior Group Velocity for an E4 Stencil

```python
import numpy as np
import matplotlib.pyplot as plt
from stencil_gen.group_velocity import interior_group_velocity

xi = np.linspace(0.01, np.pi, 500)

# E4 scheme: p=2 (half-bandwidth), nu=1 (first derivative)
profile = interior_group_velocity(p=2, nu=1, xi_array=xi)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Group velocity
axes[0].plot(xi, profile.group_velocity, label="E4 interior")
axes[0].axhline(1.0, color="k", ls="--", label="Exact")
axes[0].axhline(0.0, color="r", ls=":", alpha=0.5)
axes[0].axvline(profile.cutoff_xi, color="gray", ls=":", label=f"Cutoff = {profile.cutoff_xi:.2f}")
axes[0].set_xlabel(r"$\xi$")
axes[0].set_ylabel(r"$C(\xi)$")
axes[0].set_title("Group velocity")
axes[0].legend()

# Relative error
axes[1].semilogy(xi, np.abs(profile.gv_error))
axes[1].set_xlabel(r"$\xi$")
axes[1].set_ylabel(r"$|C - 1|$")
axes[1].set_title("Relative group velocity error")

plt.tight_layout()
plt.savefig("e4_group_velocity.png", dpi=150)
```

### 4.2 Comparing Boundary vs Interior Group Velocity

```python
import numpy as np
import matplotlib.pyplot as plt
from stencil_gen.group_velocity import (
    interior_group_velocity,
    boundary_group_velocity,
)

xi = np.linspace(0.01, np.pi, 500)

# Interior E4 profile
interior = interior_group_velocity(p=2, nu=1, xi_array=xi)

# Boundary profiles: E4 with tension kernel
boundary = boundary_group_velocity(
    p=2, q=3, nextra=0, nu=1,
    sigma=10.0, kernel="tension",
    xi_array=xi,
)

plt.figure(figsize=(10, 6))
plt.plot(xi, interior.group_velocity, "k-", lw=2, label="Interior")

for row_idx, prof in sorted(boundary.items()):
    plt.plot(xi, prof.group_velocity, "--",
             label=f"Boundary row {row_idx} (cutoff={prof.cutoff_xi:.2f})")

plt.axhline(1.0, color="gray", ls=":", alpha=0.5)
plt.axhline(0.0, color="r", ls=":", alpha=0.5)
plt.xlabel(r"$\xi$")
plt.ylabel(r"$C(\xi)$")
plt.title("Boundary vs Interior Group Velocity (E4, tension)")
plt.legend()
plt.savefig("boundary_vs_interior_gv.png", dpi=150)
```

### 4.3 Psi Sweep for Cut-Cell Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from stencil_gen.temo import E2_1
from stencil_gen.group_velocity import psi_sweep_group_velocity

xi = np.linspace(0.01, np.pi, 300)
psi_values = np.linspace(0.05, 0.95, 19)

result = psi_sweep_group_velocity(
    scheme_params=E2_1,
    psi_values=psi_values,
    alpha_values={},      # use zeros for free parameters
    xi_array=xi,
)

print(f"Worst row:          {result.worst_row}")
print(f"Worst psi:          {result.worst_psi:.3f}")
print(f"Most negative C:    {result.min_C:.4f}")
print(f"Sign reversal:      {result.has_sign_reversal}")

# Plot row 0 across psi values
fig, ax = plt.subplots(figsize=(10, 6))
for pv in psi_values[::4]:  # plot every 4th psi
    prof = result.profiles[float(pv)][0]
    ax.plot(xi, prof.group_velocity, label=f"psi={pv:.2f}")

ax.axhline(0, color="r", ls=":", alpha=0.5)
ax.set_xlabel(r"$\xi$")
ax.set_ylabel(r"$C(\xi)$")
ax.set_title("Cut-cell group velocity (row 0) vs psi")
ax.legend(fontsize=8, ncol=2)
plt.savefig("psi_sweep_gv.png", dpi=150)
```

### 4.4 2D Anisotropy Check

```python
import numpy as np
import matplotlib.pyplot as plt
from stencil_gen.group_velocity import anisotropy_profile

theta = np.linspace(0, 2 * np.pi, 360)

# Compare E2 and E4 anisotropy at xi_mag = 1.0
for p, label in [(1, "E2"), (2, "E4")]:
    result = anisotropy_profile(p=p, nu=1, theta_array=theta, xi_mag=1.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Polar plot of group speed
    ax = axes[0]
    ax = plt.subplot(121, projection="polar")
    ax.plot(result.theta, result.speed, label=f"{label}")
    ax.plot(result.theta, np.ones_like(theta), "k--", alpha=0.3, label="Exact")
    ax.set_title(f"{label} group speed (xi_mag=1.0)")
    ax.legend()

    # Angle error
    axes[1].plot(np.degrees(theta), np.degrees(result.angle_error))
    axes[1].set_xlabel("Propagation angle (deg)")
    axes[1].set_ylabel("Angle error (deg)")
    axes[1].set_title(f"{label} angle error")

    plt.tight_layout()
    plt.savefig(f"{label.lower()}_anisotropy.png", dpi=150)
```

### 4.5 GKS Stability Diagnostic

```python
import numpy as np
from stencil_gen.group_velocity import gks_group_velocity_check

# Build a differentiation matrix for E2 on N points with some boundary closure.
# (In practice, you would assemble this from your stencil pipeline.)
N = 64
h = 1.0 / N

# Interior: central difference [-1/2, 0, 1/2] / h
D = np.zeros((N, N))
for i in range(1, N - 1):
    D[i, i - 1] = -0.5 / h
    D[i, i + 1] = 0.5 / h

# Boundary closures (one-sided)
D[0, 0] = -1.0 / h
D[0, 1] = 1.0 / h
D[N - 1, N - 2] = -1.0 / h
D[N - 1, N - 1] = 1.0 / h

xi = np.linspace(0.01, np.pi, 500)
modes = gks_group_velocity_check(D, xi)

if not modes:
    print("No GKS-unstable modes found -- boundary closure is stable.")
else:
    for m in modes:
        print(f"Eigenvalue: {m.eigenvalue:.4f}")
        print(f"  Boundary wavenumber: {m.boundary_wavenumber:.3f}")
        print(f"  Group velocity:      {m.group_velocity:.4f}")
        print(f"  Outgoing (unstable): {m.is_outgoing}")
```

---

## 5. Sweep Integration

The sweep optimization pipeline (`scripts/stencil_gen/sweeps/`) consumes this
module exclusively through the thin wrappers in `sweeps/gv_objectives.py`.
Sweep scripts never call `interior_group_velocity` / `boundary_group_velocity`
/ `psi_sweep_group_velocity` / `gks_group_velocity_check` directly — all access
goes through one of the helpers below so the `feasible-then-minimize` contract
and the bit-exact `(param, gv_error)` self-consistency contract (see
`docs/sweeps_reference.md` §2.9) stay consistent across every sweep.

### 5.1 Objective Helpers in `sweeps/gv_objectives.py`

| Helper | Wraps | Returns | Used by |
|---|---|---|---|
| `interior_gv_error_max(p, nu, n_xi)` | `interior_group_velocity` | `float` — max `|gv_error|` over `xi` | — (reserved for future interior-only sweeps) |
| `interior_cutoff_fraction(p, nu, n_xi)` | `interior_group_velocity` | `float` — `cutoff_xi / pi` (1.0 = ideal) | — (reserved) |
| `boundary_gv_error_max(p, q, nextra, nu, sigma, kernel, n_xi)` | `boundary_group_velocity` | `float` — max `|gv_error|` across all RBF boundary rows | `tension_sweep`, `epsilon_sweep`, `footprint_sweep`, `gv_stability_pareto` |
| `cutcell_gv_min_C(scheme_params, psi_values, alpha_values, n_xi)` | `psi_sweep_group_velocity` | `(float, bool)` — `(min_C, has_sign_reversal)` | — (reserved for the deferred 40.5f cut-cell GV sweep) |
| `gv_score_from_matrix(D, n_xi)` | `group_velocity_exact_nonuniform` + `group_velocity_error` | `dict` with `max_gv_error` / `min_cutoff_xi` | `tension_penalty_sweep` (avoids rebuilding `D`) |
| `print_gks_advisory(D, *, label, n_xi)` | `gks_group_velocity_check` | `int` — outgoing mode count (advisory only) | `tension_sweep`, `epsilon_sweep` (`--check-gks`) |

**Contract:** these helpers are *secondary* objectives only. Stability
(`stability_eigenvalue < STABILITY_TOL`) is always the hard feasibility gate;
GV error is minimized only among the feasible set. This mirrors the
feasible-then-minimize pattern already established in
`tension_penalty_sweep.py`.

**Default `n_xi`:** all helpers use `np.linspace(1e-6, pi, 200)` for the
wavenumber grid. This is enough for the ~1e-6 quadrature noise floor that the
bit-exact regression tests in `tests/test_sweep_gv_objectives.py` rely on; see
`_default_xi` in `gv_objectives.py`.

**Deterministic output:** `boundary_gv_error_max` is a pure numpy pipeline with
no RNG or cached state, so two consecutive calls at identical arguments return
bit-identical floats. This is load-bearing for the 40.8f / 40.8g bit-exact
gates — any helper added to this module must preserve that property.

### 5.2 GKS Check is Advisory Only

`gks_group_velocity_check` is exposed to sweeps through
`print_gks_advisory` and surfaced via the `--check-gks` flag on
`tension_sweep` and `epsilon_sweep`. It is **necessary not sufficient** for
boundary instability (see the test docstring at
`tests/test_group_velocity.py:1015-1017`): a clean advisory does not prove
the scheme is GKS-stable, but any outgoing-mode warning it emits is a real
finding that should be investigated.

Consequently the sweep integration treats the GKS check as a *diagnostic*,
never as a feasibility gate:

- It runs **after** the stability-optimum and the `stable_at` cross-grid
  list have already been built, so it cannot mutate the optimum by
  construction, not just by convention.
- It does not alter `known_values.json`.
- It does not change the exit code of the sweep.
- Any outgoing modes print as one-line `WARNING:` advisories with
  `xi`, `eigenvalue`, and `group_velocity` fields.

If you want to use the GKS check as a hard gate in your own tooling, call
`gks_group_velocity_check` directly from this module and apply your own
policy — do not add a feasibility branch to the sweep helpers.

For a **rigorous** GKS stability test (necessary *and* sufficient for the 1D
reduction), use `kreiss_stability_check` from `stencil_gen.gks_kreiss` (see
`docs/brady2d_stability_reference.md`). The Kreiss determinant test sweeps
`s` in the right half-plane, refines witnesses, and classifies imaginary-axis
perturbations per Trefethen 1983 pp. 206-207. It is used as Layer 2 (L2) of
the Brady-Livescu 2D stability pipeline.

### 5.3 `known_values.json` Cross-Reference

The persisted `*.gv_error` fields, `tension_gv` / `{kernel}_gv` /
`tension_penalty_gv` entries, and `footprint.E4_nextra{nx}_tension_gv`
entries are all computed through these helpers (see
`docs/sweeps_reference.md` §2.9 and §3 for the schema). The primary
`{primary}.gv_error` fields (`tension`, `gaussian`, `multiquadric` via
`boundary_gv_error_max`; `tension_penalty` via `gv_score_from_matrix`; and
the footprint `E4_nextra{nx}_tension_{N}.gv_error` via
`boundary_gv_error_max`) are contractually bit-exact self-consistent with
their paired `sigma` / `epsilon` / `(sigma, gamma)` field: rebuilding `D`
at the persisted parameter and re-running the paired helper returns the
stored `gv_error` to the full 13 significant digits of float64. The
regression tests in `tests/test_phs.py::TestRegressionGV` and
`tests/test_sweep_gv_objectives.py::test_*_bit_exact_at_persisted_sigma`
gate this contract.

---

## 6. Key Mathematical Notes

### Modified Wavenumber and Dispersion Relation

For a stencil with weights `w_j` at offsets `d_j` from the evaluation point, the modified wavenumber is:

    kappa*(xi) = sum_j w_j exp(i * d_j * xi)

For the exact first derivative `d/dx`, `kappa*(xi) = i*xi`, giving `Im(kappa*) = xi` (perfect dispersion). Any finite-difference stencil introduces truncation error in `kappa*`.

### Sign Convention for the Advection Equation

The module adopts the convention `u_t + u_x = 0` (rightward propagation at unit speed). The semi-discrete form is `du/dt = -D*u` where `D` approximates `d/dx`. The physical group velocity is `C = +1`. A computed `C(xi) < 0` means spurious backward propagation at wavenumber `xi`.

### Relationship to Eigenvalue Stability

The eigenvalues of `-D` (with boundary conditions applied) control time-stepping stability. The modified wavenumber `kappa*` of the interior stencil gives the imaginary parts of these eigenvalues on a periodic domain. Group velocity analysis complements eigenvalue analysis by explaining *where* energy goes -- an eigenvalue can be neutrally stable (purely imaginary) while its associated mode's group velocity directs energy toward a boundary that reflects it, leading to algebraic growth or GKS instability.

### (2p+1)x Error Amplification

For a `2p`-order interior stencil, the phase velocity error is `O(xi^(2p))` at small `xi`. Because the group velocity is the *derivative* of the dispersion relation:

    C(xi) = d(Im(kappa*))/d(xi)

the leading error term gains a factor of `(2p+1)` compared to the phase velocity error. Concretely, if `Im(kappa*) = xi + c * xi^(2p+1)`, then `C = 1 + c*(2p+1)*xi^(2p)`. This means group velocity error is `(2p+1)` times larger than the corresponding phase velocity error at any given `xi`.

### Cutoff Wavenumber

The **cutoff wavenumber** `cutoff_xi` is defined as the first `xi` beyond which the group velocity `C(xi)` stays permanently non-positive. Waves with `xi > cutoff_xi` propagate in the wrong direction. For E2 (`p=1`), the interior cutoff is `xi = pi/2` (where `cos(pi/2) = 0`). Higher-order schemes push the cutoff closer to `pi`.

Boundary stencils typically have a lower cutoff than the interior, meaning they produce backward-propagating energy at a wider range of wavenumbers. This is a key mechanism for boundary instability.

### GKS Instability Mechanism

Gustafsson, Kreiss, and Sundstrom (1972) showed that IBVP stability requires checking whether boundary-reflected waves carry energy into the domain. Trefethen (1983) connected this to group velocity: a boundary mode is GKS-unstable if:

1. The eigenvalue is nearly neutral (small damping).
2. The mode is localized near the boundary.
3. The interior stencil's group velocity at the mode's dominant wavenumber directs energy **from the boundary into the domain**.

The `gks_group_velocity_check` function automates this three-step diagnostic
as a heuristic. The rigorous formulation is the Kreiss determinant test in
`stencil_gen.gks_kreiss`: for each `s` with `Re(s) >= 0`, it assembles the
Kreiss matrix from admissible roots (`|kappa| < 1`) and checks whether
`sigma_min(M(s))` vanishes, confirming a genuine GKS violation.

### Non-Uniform Offsets (Cut Cells)

Cut-cell stencils have a wall point at fractional distance `psi` from the nearest grid point. The offsets become non-integer: the wall contributes offset `-(psi + i)` for boundary row `i`, while regular grid points contribute integer offsets `j - i`. The `_nonuniform` variants of modified wavenumber and group velocity handle these fractional offsets correctly.

---

## References

- Trefethen, L.N. "Group velocity in finite difference schemes." *SIAM Review*, 24(2):113-136, 1982.
- Trefethen, L.N. "Stability and group velocity." *Numerische Mathematik*, 41:1-29, 1983 (GKS connection).
- Gustafsson, B., Kreiss, H.-O., and Sundstrom, A. "Stability theory of difference approximations for mixed initial boundary value problems." *Mathematics of Computation*, 26(119):649-686, 1972.
- Brady, P.T. and Livescu, D. "High-order, stable, and conservative boundary schemes for central and compact finite differences." *Computers & Fluids*, 183:84-101, 2019.
