# Stencil Generation Pipeline Reference

## 1. Pipeline Overview

The `stencil_gen` pipeline derives finite difference stencil coefficients symbolically
using SymPy and generates C++ source files for the SHOCCS solver. The end-to-end flow is:

```
SchemeParams (p, q, s, nextra, nu)
    |
    v
Interior derivation (Taylor matching) --> exact rational coefficients
    |
    v
Uniform boundary derivation (underdetermined Taylor systems + free alpha params)
    |
    v
Discrete-conservation (telescoping/flux) constraints --> eliminate last-row placeholders, compute weights
    |
    v
TEMO cut-cell extension (psi-parameterized stencils for embedded boundaries)
    |
    v
Code generation --> C++ .cpp and .t.cpp files for src/stencils/
```

There are two main entry points for the full pipeline:

- **`derive_cut_cell_mathematica()`** -- the singularity-free path (recommended for
  schemes with `zeros`). Keeps alpha as free constants throughout, matching the
  Mathematica workflow in `explicitr-E4d1.nb`.

- **`derive_cut_cell_scheme()`** -- the general-purpose entry point. Handles both
  conserved and non-conserved paths, zero-constrained schemes (E4_1), and Neumann
  stencils (nu=2).

Generated C++ output lands in `scripts/stencil_gen/output/` and replaces files in
`src/stencils/`.


## 2. Module Map

### Core Derivation Modules (dependency order)

| Module | Purpose | Key Entry Points |
|--------|---------|-----------------|
| `_util.py` | Shared linear solver wrapper | `solve_linear(A, b, unknowns)` |
| `taylor_system.py` | Vandermonde-like Taylor matching systems | `build_taylor_system(i, t, q, nu)` |
| `interior.py` | Interior stencil coefficients | `derive_interior(s, p, nu)`, `full_gamma_array()`, `full_delta_array()` |
| `boundary.py` | Boundary stencil derivation (underdetermined) | `solve_boundary_row()`, `derive_boundary(p, nu, s)` |
| `conservation.py` | Discrete-conservation (telescoping/flux) constraints | `build_conservation_system()`, `solve_conservation()` |
| `temo.py` | TEMO cut-cell extension + uniform boundary for TEMO | `derive_cut_cell_scheme()`, `derive_cut_cell_mathematica()`, `derive_e2_uniform_boundary()`, `derive_uniform_boundary_for_temo()` |

### Code Generation

| Module | Purpose | Key Entry Points |
|--------|---------|-----------------|
| `printer.py` | SymPy-to-C++ expression printer | `StencilCodePrinter`, `build_symbol_map()` |
| `codegen.py` | Full C++ file generation | `generate_stencil_cpp(spec)`, `generate_test_cpp(spec, test_cases)`, `generate_interior_method()`, `generate_nbs_method()` |

### Analysis Modules

| Module | Purpose | Key Entry Points |
|--------|---------|-----------------|
| `phs.py` | PHS/RBF stencil weights + stability analysis | `phs_stencil_weights()`, `build_diff_matrix_rbf()`, `stability_eigenvalue()`, `cut_cell_weights()` |
| `group_velocity.py` | Dispersion/group velocity analysis | `modified_wavenumber()`, `group_velocity_exact()`, `phase_velocity()`, `group_velocity_error()` |

### Sweep Scripts (`sweeps/` package)

| Module | Purpose |
|--------|---------|
| `epsilon_sweep.py` | Gaussian/Multiquadric epsilon parameter sweep |
| `tension_sweep.py` | Tension spline sigma parameter sweep |
| `tension_penalty_sweep.py` | Joint (sigma, gamma) tension + conservation penalty sweep |
| `mixed_epsilon_sweep.py` | Per-row epsilon sweep for RBF stencils |
| `footprint_sweep.py` | Stencil footprint (nextra) sweep |
| `comparison.py` | Multi-method comparison tables |
| `alpha_extraction.py` | Extract boundary alphas at optimal epsilon |
| `_common.py` | Shared helpers: `SweepResult`, `print_table()`, `known_values.json` I/O |


## 3. Scheme Parameters

### SchemeParams

Defined in `temo.py`, `SchemeParams` encapsulates the five parameters that uniquely
define a stencil scheme:

```python
@dataclass(frozen=True)
class SchemeParams:
    p: int          # Interior half-width (RHS bandwidth)
    q: int          # Boundary accuracy order
    s: int          # LHS half-width (0=explicit, 1=tridiagonal compact)
    nextra: int     # Extra rows/columns for numerical optimization
    nu: int         # Derivative order (1 or 2)
    zeros: tuple[int, ...] = ()  # Alpha indices forced to zero post-hoc
```

### Pre-defined Schemes

| Name | p | q | s | nextra | nu | zeros | Notes |
|------|---|---|---|--------|----|-------|-------|
| `E2_1` | 1 | 1 | 0 | 1 | 1 | -- | 2nd-order explicit, 1st derivative |
| `E2_2` | 1 | 1 | 0 | 0 | 2 | -- | 2nd-order explicit, 2nd derivative |
| `E4_1` | 2 | 3 | 0 | 0 | 1 | (3,4) | 4th-order explicit, 1st derivative |
| `E4_2` | 2 | 3 | 0 | 0 | 2 | -- | 4th-order explicit, 2nd derivative |

### Dimension Formulas

`compute_dimensions(p, q, s, nextra, nu)` returns a `Dimensions(r, t, R, T, X)` named tuple:

**For 1st derivatives (nu=1):**
```
t = p + q + 1 + nextra     (boundary stencil width)
r = q + 1 + nextra         (number of uniform boundary rows)
R = r + 1                  (cut-cell rows = uniform rows + wall row)
T = t + 1                  (cut-cell columns = uniform + wall column)
X = 0                      (no Neumann extra rows)
```

**For 2nd derivatives (nu=2):**
```
t = p + 2 + nextra
r = p + 1 + nextra
r_eff = r - 1              (last uniform row overlaps first interior)
R = r_eff + 1
T = t + 1
X = R                      (Neumann extra rows)
```

### StencilSpec (interior.py)

For interior stencils only:

```python
@dataclass(frozen=True)
class StencilSpec:
    s: int   # LHS half-bandwidth
    p: int   # RHS half-bandwidth
    nu: int  # derivative order

    order = 2 * (p + s)          # formal order of accuracy
    n_unknowns = p + s           # number of independent coefficients
    scheme_name: E2_1, T4_1, etc.
```


## 4. Pipeline Stages

### Stage 1: Interior Derivation (`interior.py`)

**Input:** `(s, p, nu)` -- LHS bandwidth, RHS bandwidth, derivative order.

**Method:** Taylor series matching. For an interior stencil centered at grid point i,
the coefficients must reproduce derivatives exactly for polynomials up to degree
`2(p+s)-1`. The resulting `2(p+s)` conditions (after eliminating trivially-satisfied
equations via symmetry) produce a square system of `p+s` unknowns.

**Output:** `InteriorCoefficients` containing:
- `gamma[j]` for j=1..p (RHS coefficients; negative indices from anti/symmetry)
- `delta[k]` for k=1..s (LHS coefficients; negative indices from symmetry)

**Example:** E4_1 has `p=2, s=0` giving `gamma = {1: 2/3, 2: -1/12}`, expanded to
`[-1/12, 2/3, 0, -2/3, 1/12]` via antisymmetry.

### Stage 2: Boundary Derivation (`boundary.py`)

**Input:** `(p, nu, s)` -- same as interior, plus scheme determines boundary
dimensions `r = 2p-1` rows, `t = r+p` columns, order `q = 2(p+s)-1`.

**Method:** Each boundary row `i` (0..r-1) has a `(q+1) x t` Vandermonde system
from Taylor matching. Since `t > q+1`, there are `n_free = t - (q+1)` free parameters
per row. These are represented by `alpha` symbols.

The last row's free parameters are placeholder `phi` symbols, resolved in the
conservation step.

**Output:** `BoundaryResult` containing `r` `BoundaryRow` objects (coefficients as
symbolic expressions in alpha parameters), interior coefficients, and the global
list of free alpha symbols.

### Stage 3: Conservation (`conservation.py`)

**Input:** Boundary rows, interior coefficients, scheme dimensions.

**Method:** The discrete-conservation (telescoping/flux) condition requires that
weighted column sums of the full operator vanish for interior columns, and equal
-1 for column 0. This yields `t-1` linear equations in the conservation
(quadrature) weights `w_0..w_{r-1}` and the last row's `phi` placeholders.

The key challenge is **bilinear terms** `w_{r-1} * phi_k` (product of two unknowns).
These are linearized via the theta-substitution trick: define `theta_k = w_{r-1} * phi_k`,
solve the resulting linear system, then recover `phi_k = theta_k / w_{r-1}`.

**Output:** Solution dict mapping each `w_i` and `phi_j` to expressions in alpha
parameters, plus updated boundary rows with placeholders resolved.

### Stage 4: TEMO Cut-Cell Extension (`temo.py`)

**Input:** Uniform boundary matrix `B_u`, interior coefficients, scheme params, `psi` symbol.

**Method:** The TEMO (Truncation Error Matching Optimization) procedure from
Brady & Livescu (2021) extends uniform boundary stencils to handle cut cells.
The cut-cell stencil has `R = r+1` rows and `T = t+1` columns (extra column for
the wall point at fractional distance `psi` from the boundary).

Key sub-steps:
1. **Degenerate stencil** (`build_degenerate_stencil`): construct a base cut-cell
   stencil that reduces to the uniform stencil at `psi=1`.
2. **TEMO Vandermonde system** (`build_temo_vandermonde`): Taylor matching with
   non-integer grid spacing (psi-shifted wall point).
3. **Row solving** (`solve_temo_row` or `solve_temo_row_polynomial`): solve each
   cut-cell row, finding coefficients as rational functions of `psi`.
4. **Cut-cell conservation** (`build_cut_cell_conservation_system`,
   `solve_cut_cell_conservation`): enforce discrete conservation (the telescoping/flux condition) on the cut-cell operator.
5. **Assembly** (`assemble_cut_cell_result`): package floating, Dirichlet, and
   Neumann variants into `CutCellResult`.

For schemes with `zeros` (like E4_1), the **Mathematica workflow**
(`derive_cut_cell_mathematica`) is preferred:
1. Uniform boundary with zeros applied
2. Uniform conservation with weight constraint
3. Base scheme construction
4. Cut-cell with free gammas
5. Cut-cell conservation
6. Blend remaining free gammas
7. Assembly

**Output:** `CutCellResult` with:
- `floating`: R x T matrix (psi-dependent symbolic entries)
- `dirichlet`: (R-1) x T matrix (row 0 removed for Dirichlet BC)
- `neumann`: R x T matrix (only for nu=2)
- `eta`: Neumann eta coefficients (only for nu=2)
- `dims`, `alpha_symbols`, `weights`, `conservation_subs`

### Stage 5: Code Generation (`codegen.py`)

**Input:** `StencilGenSpec` containing all coefficients and dimensions.

**Method:** Generates complete C++ source files matching the patterns in `src/stencils/`:
1. `_emit_header()` -- includes, namespace open
2. `_emit_struct_preamble()` -- struct definition, constants P/R/T/X, member arrays, constructors
3. `_emit_query_methods()` -- `query_max()`, `query()`, `query_interp()`
4. `_emit_interior_method()` -- rational coefficients with baked-in `h` division
5. `_emit_nbs_dispatcher()` -- `nbs()` switch on BC type
6. `_emit_nbs_methods()` -- `nbs_floating()`, `nbs_dirichlet()`, `nbs_neumann()` with CSE
7. `_emit_factory()` -- `make_<name>()` factory function

The printer (`StencilCodePrinter`) handles:
- `Pow(x,2)` -> `x * x` (not `std::pow`)
- `Rational(p,q)` -> `p.0 / q`
- Alpha symbols -> array indexing (`alpha_0` -> `alpha[0]`)

Test file generation (`generate_test_cpp`) creates Catch2 test files with:
- Lua configuration block
- Numerical evaluation of symbolic coefficients at test points
- `REQUIRE_THAT` with `Approx` matcher


## 5. Analysis Modules

### phs.py -- RBF/PHS Stencil Weights and Stability

Computes finite difference weights using Polyharmonic Spline (PHS) and other RBF
kernels. The augmented system is:

```
[Phi  P] [lambda]   [d_Phi]
[P'  0] [mu    ] = [d_P  ]
```

where Phi is the RBF kernel matrix, P enforces polynomial reproduction, and
lambda are the FD weights.

**Key functions:**

| Function | Purpose |
|----------|---------|
| `phs_stencil_weights(points, x_eval, nu, k, q)` | PHS+polynomial FD weights (exact rational via SymPy) |
| `uniform_interior_weights(p, nu, k, q)` | Interior weights on uniform grid |
| `uniform_boundary_weights(i, t, nu, k, q)` | Boundary row weights on uniform grid |
| `build_diff_matrix_rbf(n, p, q, nu, nextra, kernel, ...)` | Full n x n differentiation matrix |
| `build_diff_matrix_rbf_penalty(n, p, q, nu, nextra, ...)` | Diff matrix with conservation penalty |
| `build_diff_matrix_mixed_epsilon(n, p, q, nu, nextra, ...)` | Diff matrix with per-row epsilon values |
| `stability_eigenvalue(n, p, q, nu, nextra, ...)` | Max real eigenvalue (stability indicator) |
| `stability_eigenvalue_from_matrix(D)` | Stability from pre-built matrix |
| `max_real_eigenvalue(D)` | Raw max Re(lambda) of matrix D |
| `cut_cell_weights(points, x_eval, nu, k, q)` | Weights for cut-cell (non-uniform) grids |

**Supported kernels:** `"phs"` (r^(2k-1)), `"gaussian"` (exp(-eps^2 r^2)),
`"multiquadric"` (sqrt(1 + eps^2 r^2)), `"tension"` (sigma|r| - 1 + exp(-sigma|r|)).

### group_velocity.py -- Dispersion Analysis

Analyzes numerical dispersion properties of stencils via modified wavenumber theory.
For `u_t + u_x = 0` semi-discretized as `du/dt = -D*u`:

**Key functions:**

| Function | Purpose |
|----------|---------|
| `modified_wavenumber(weights, i_eval, nodes, xi)` | kappa*(xi) for uniform grids |
| `modified_wavenumber_nonuniform(weights, offsets, xi)` | kappa*(xi) for cut-cell grids |
| `group_velocity_exact(weights, i_eval, nodes, xi)` | C(xi) = d(Im(kappa*))/d(xi) analytically |
| `group_velocity_exact_nonuniform(weights, offsets, xi)` | C(xi) for non-uniform offsets |
| `group_velocity_numerical(kappa_star, xi)` | C(xi) via numpy.gradient |
| `phase_velocity(kappa_star, xi)` | c(xi) = Im(kappa*)/xi |
| `group_velocity_error(C, C_exact)` | Relative error (C - C_exact)/C_exact |

Also provides 2D analysis via `GroupVelocity2DResult` and `AnisotropyResult` for
tensor-product operators.

### sweeps/ -- Parameter Space Exploration

Research tools (not regression tests) that explore parameter spaces to discover
optimal values for RBF-augmented stencils.

**Workflow:**
1. Run a sweep script to explore a parameter space
2. Results are printed and optionally written to `sweeps/known_values.json`
3. Regression tests in `tests/test_phs.py` load from `known_values.json` and verify
   stability at discovered values

**Available sweeps:**

| Command | What it explores |
|---------|-----------------|
| `epsilon --scheme E2/E4` | Optimal Gaussian/Multiquadric epsilon |
| `tension --scheme E2/E4` | Optimal tension spline sigma |
| `tension-penalty --scheme E2/E4` | Joint (sigma, gamma) for tension + conservation penalty |
| `mixed-epsilon --scheme E2/E4` | Per-row epsilon optimization |
| `footprint` | Effect of nextra (stencil footprint width) |
| `comparison` | Multi-method comparison table |
| `alpha --scheme E2/E4` | Extract boundary alphas at optimal epsilon |
| `all [--quick]` | Run all sweeps (--quick for reduced resolution) |


## 6. Test Structure

All tests are in `scripts/stencil_gen/tests/` and use pytest.

### Test File Coverage

| Test File | Module(s) Covered | Speed |
|-----------|-------------------|-------|
| `test_interior.py` | `interior.py` | Fast |
| `test_boundary.py` | `boundary.py`, `conservation.py`, `taylor_system.py` | Fast (E4u, E6u, E8u pipelines cached via module fixtures) |
| `test_temo.py` | `temo.py` (dimensions, uniform boundary, degenerate stencil, psi field, TEMO vandermonde, cut-cell construction, Neumann, assembly, E2 integration) | Mostly fast |
| `test_e4_cut_cell.py` | `temo.py` (E4 uniform boundary, zero-constrained cut-cell, TEMO construction, conservation, code generation, Mathematica workflow) | Mix of fast and slow (`@pytest.mark.slow`) |
| `test_phs.py` | `phs.py` (PHS core, interior/boundary weights, diff matrices, stability, Gaussian/tension kernels, conservation penalty, modified wavenumber) | Mostly fast |
| `test_group_velocity.py` | `group_velocity.py` (core, interior, boundary, cut-cell, 2D, anisotropy, GKS diagnostic) | Fast |
| `test_codegen.py` | `codegen.py`, `printer.py` (CSE, interior/nbs methods, full struct, test file generation) | Fast |
| `test_codegen_e4u.py` | `codegen.py` (E4u end-to-end: interior coefficients, floating/Dirichlet numerical, full pipeline, compilation check) | Fast |
| `test_printer.py` | `printer.py` (Pow, Rational, Integer, Symbol formatting) | Fast |
| `test_eval_e2_1.py` | `temo.py` (E2_1 numerical evaluation at specific parameter values) | Fast |

### conftest.py Fixtures

- `assert_taylor_accuracy` (session) -- verifies Taylor moment conditions on B_u matrices
- `e2_1_uniform`, `e2_2_uniform` (module) -- cached uniform boundary results
- `e4u_pipeline`, `e6u_pipeline`, `e8u_pipeline` (module) -- cached full boundary+conservation pipeline results
- `run_pipeline(p, nu, s)` -- helper to run derive_boundary + conservation
- `--run-slow` CLI option -- enables `@pytest.mark.slow` tests

### Slow Tests

Tests marked `@pytest.mark.slow` are in `test_e4_cut_cell.py` and include:
- `TestE4CutCellConservationSolution`
- `TestE4TEMOConstruction`
- `TestE4CodeGeneration`
- `TestE4TestFileGeneration`
- `TestDeriveCutCellScheme`
- `TestE4CutCellSchemeWithZeros`
- `TestPolynomialFullStencil`
- `TestMathematicaWorkflow`
- Various standalone conservation tests

These are slow because they run the full TEMO pipeline for E4 (4th-order), which
involves large symbolic linear algebra.


## 7. Key Performance Rules

### SYMPY_CACHE_SIZE=50000

**This is essential.** The default SymPy cache size of 1000 causes severe slowdowns
with the large symbolic expressions in boundary and TEMO derivation. Always set:

```bash
SYMPY_CACHE_SIZE=50000 uv run pytest tests/ -x -q
```

### What Is Slow and Why

1. **E4 TEMO pipeline** -- the cut-cell construction for 4th-order schemes involves
   solving in `QQ(psi, alpha_0, alpha_1, ...)` polynomial fields, which is
   computationally expensive in SymPy.

2. **`solve_in_field()`** -- solving Vandermonde systems over polynomial fraction
   fields using `DomainMatrix` is the bottleneck in the TEMO pipeline.

3. **`cse()` on large expressions** -- common subexpression elimination for code
   generation can be slow for psi-dependent cut-cell coefficients.

4. **Conservation with bilinear terms** -- the theta-substitution linearization
   and subsequent `cancel()` calls are expensive for large symbolic expressions.

### What Is Fast

- Interior derivation (small exact-rational systems)
- E2 boundary and TEMO (small stencils)
- Group velocity analysis (numpy-based)
- PHS weight computation (small symbolic systems)
- Stability eigenvalue computation (numpy eigenvalue)
- All printer and codegen tests (string manipulation)

### Solver Preferences

- Use `linear_eq_to_matrix` + `linsolve` for linear systems (not `sympy.solve`).
- For bilinear terms, linearize first with theta-substitution.
- Use `cancel()` liberally to keep expressions manageable.
- Use `DomainMatrix` over `QQ(psi)` for TEMO field arithmetic.


## 8. Common Tasks

### Derive a New Stencil Scheme

1. Define `SchemeParams` with appropriate (p, q, s, nextra, nu, zeros).

2. For first derivatives (nu=1), call the high-level entry point:
   ```python
   from sympy import Symbol
   from stencil_gen.temo import SchemeParams, derive_cut_cell_scheme

   scheme = SchemeParams(p=3, q=5, s=0, nextra=0, nu=1)
   psi = Symbol("psi")
   result = derive_cut_cell_scheme(scheme, psi)
   ```

3. For schemes with zeros (like E4_1), use the Mathematica workflow:
   ```python
   from stencil_gen.temo import E4_1, derive_cut_cell_mathematica

   psi = Symbol("psi")
   result = derive_cut_cell_mathematica(E4_1, psi)
   ```

4. To get just the uniform boundary (no cut-cell):
   ```python
   from stencil_gen.temo import derive_uniform_boundary_for_temo

   uniform = derive_uniform_boundary_for_temo(scheme)
   # uniform.B_u is the boundary matrix, uniform.alpha_symbols are free params
   ```

5. To get just the interior coefficients:
   ```python
   from stencil_gen.interior import derive_interior, full_gamma_array

   coeffs = derive_interior(s=0, p=3, nu=1)
   gamma = full_gamma_array(coeffs)  # length 2*p+1 list
   ```

### Check Stability of Existing Scheme

```python
from stencil_gen.phs import stability_eigenvalue

# For a uniform PHS stencil
se = stability_eigenvalue(
    n=40, p=1, q=1, nu=1, nextra=1,
    kernel="phs", k=2,
)
print(f"Max real eigenvalue: {se}")
print(f"Stable: {se < 1e-10}")

# For a prebuilt differentiation matrix
from stencil_gen.phs import stability_eigenvalue_from_matrix
se = stability_eigenvalue_from_matrix(D)
```

### Run Group Velocity Analysis

```python
import numpy as np
from stencil_gen.group_velocity import (
    modified_wavenumber,
    group_velocity_exact,
    phase_velocity,
    group_velocity_error,
)

# E2 interior stencil: [-1/2, 0, 1/2]
weights = [-0.5, 0.0, 0.5]
xi = np.linspace(0, np.pi, 200)

kappa = modified_wavenumber(weights, i_eval=1, node_indices=[0, 1, 2], xi_array=xi)
C = group_velocity_exact(weights, i_eval=1, node_indices=[0, 1, 2], xi_array=xi)
c = phase_velocity(kappa, xi)
err = group_velocity_error(C)
```

### Generate C++ Code

```python
from stencil_gen.codegen import StencilGenSpec, generate_stencil_cpp, generate_test_cpp

spec = StencilGenSpec(
    name="polyE4_1",
    P=2, R=5, T=7, X=0,
    derivative_order=1,
    is_uniform=False,
    param_arrays={"alpha": 2},
    interior_coeffs=[...],       # from derive_interior
    floating_coeffs=[...],       # R*T symbolic expressions
    dirichlet_coeffs=[...],      # R*T symbolic expressions (row 0 skipped internally)
)

cpp_code = generate_stencil_cpp(spec)
with open("output/polyE4_1.cpp", "w") as f:
    f.write(cpp_code)
```

### Run Parameter Sweeps

```bash
# From scripts/stencil_gen/

# Single sweep
uv run python -m sweeps epsilon --scheme E2 --kernel gaussian
uv run python -m sweeps tension --scheme E4
uv run python -m sweeps tension-penalty --scheme E4

# Multi-method comparison
uv run python -m sweeps comparison

# All sweeps (quick mode for fast verification)
uv run python -m sweeps all --quick

# Save optimal values to known_values.json
uv run python -m sweeps epsilon --scheme E2 --update-known-values
```

### Run Tests

```bash
cd scripts/stencil_gen

# All fast tests
SYMPY_CACHE_SIZE=50000 uv run pytest tests/ -x -q

# Skip slow full-pipeline E4 tests
SYMPY_CACHE_SIZE=50000 uv run pytest tests/ -x -q \
    -k "not TestMathematicaWorkflow and not TestPolynomialFullStencil and not TestE4CodeGeneration"

# Include slow tests
SYMPY_CACHE_SIZE=50000 uv run pytest tests/ -x -q --run-slow

# Single test file
SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_interior.py -x -q

# Single test class
SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_temo.py::TestDimensions -x -q
```
