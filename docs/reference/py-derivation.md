# Python Stencil Derivation Framework (`scripts/stencil_gen/stencil_gen/`)

> **Maturity:** mature · **Audited:** 2026-05-29 · See [Capability Audit](../CAPABILITY_AUDIT.md) · [Onboarding](../ONBOARDING.md)

## Purpose
This is the SymPy-based symbolic pipeline that *derives* the finite-difference stencil coefficients the C++ solver uses — interior circulant coefficients, uniform boundary closures, and psi-parameterized cut-cell boundary rows — and emits matching C++ source for `src/stencils/`. It implements the **TEMO** (Truncation Error Matching Optimization) cut-cell method from Brady & Livescu (2021): exact-rational interior coefficients by Taylor matching, underdetermined boundary rows parameterized by free `alpha` constants, discrete-conservation (telescoping/flux) constraints, and a psi-blended cut-cell extension for embedded boundaries. A separate **numeric** arm (`phs.py`) computes PHS/RBF/tension-spline weights and stability eigenvalues that drive the parameter sweeps and optimizers. The Python side is already heavily documented; this doc is an *orientation map* — for depth, follow the cross-links rather than re-reading here.

The wider `scripts/stencil_gen/` tree also contains the sweep/optimizer/stability-analysis stack (`sweeps/`, `optimizer.py`, `bo.py`, `group_velocity.py`, `brady2d_stability.py`). Those are **downstream consumers** of this subsystem and have their own reference docs (see [Related docs](#related-docs)); they are out of scope here.

## Where it lives
| File | Role |
| --- | --- |
| `temo.py` (3305 lines) | The heart. Both cut-cell entry points, the `SchemeParams` catalog, dimension formulas, uniform-boundary builders, cut-cell + conservation + Neumann construction, `CutCellResult`. |
| `interior.py` | Exact-rational interior coefficients via Taylor matching: `derive_interior`, `full_gamma_array`, `full_delta_array`, `StencilSpec`, `InteriorCoefficients`. |
| `boundary.py` | Underdetermined boundary-row solver — coefficients as functions of free `alpha` symbols: `derive_boundary`, `solve_boundary_row`, `BoundaryRow`/`BoundaryResult`. |
| `taylor_system.py` | Builds the Vandermonde-like Taylor-matching linear system (`build_taylor_system`). Foundation for `interior`/`boundary`. |
| `conservation.py` | Discrete-conservation (telescoping/flux) constraint build/solve with theta-linearization of bilinear terms: `build_conservation_system`, `solve_conservation`, `_interior_contribution`. |
| `_util.py` | `solve_linear(A, b, unknowns)` — the project-standard `linsolve` + `cancel` wrapper used across the core. |
| `codegen.py` | C++ `.cpp`/`.t.cpp` emission: `StencilGenSpec`, `generate_stencil_cpp`, `generate_test_cpp`, `compute_test_values`. **Invoked only from tests** — no automated driver. |
| `printer.py` | SymPy→C++ expression printer matching hand-written stencil style: `StencilCodePrinter`, `build_symbol_map`. |
| `phs.py` | Numeric PHS/RBF/tension-spline weights, differentiation-matrix builders, stability eigenvalues. The numeric arm that feeds `sweeps/`/`optimizer.py`. Contains dead B-spline code (see [Maturity](#maturity--known-gaps)). |
| `cpp_bridge.py` | Python→C++ solver driver: renders a Brady-Livescu Lua config and runs the compiled `shoccs` binary for end-to-end validation. |
| `__main__.py` | CLI **stub** — only `list` works; `generate`/`validate` print `TODO`. NOT a real driver (see Gotchas). |
| `output/E4_1.cpp` | Generated artifact. Shares its regen commit with `src/stencils/E4_1.cpp` but has since diverged (src got hand-edits) — concrete evidence of the manual copy workflow. |

## Public API / entry points
`__init__.py` exports nothing; import modules by absolute path (`from stencil_gen.temo import ...`).

### Cut-cell derivation (single production entry point — read this carefully)
```python
# temo.py
derive_cut_cell_scheme(scheme, psi, alpha_symbols=None, conserve=True) -> CutCellResult
derive_cut_cell_paper(scheme, psi, alpha_symbols=None) -> CutCellResult
```
`derive_cut_cell_scheme` dispatches internally on `scheme.zeros`, so calling it alone is always correct; production callers (`group_velocity.run_psi_sweep`, `cutcell_eval.make_cutcell_evaluator`) additionally branch on `scheme.zeros` themselves to call `derive_cut_cell_paper` directly for zeros-schemes — the same dispatch, just inlined at the call site:
- Schemes **with** a non-empty `zeros` pattern (currently **E4_1**) route to **`derive_cut_cell_paper`** — a direct symbolic implementation of the published Brady & Livescu (2021, JCP 426:109794) Sec. 3.2/3.3 construction: a 5x7 block with row 0 evaluated at the wall, rows 1..r-1 centered at fluid nodes, a wall-free transition row, the paper blend rule `alpha_ij = psi*alphau_ij + (1-psi)*alphau_{i-1,j-1}`, and a linear (t+1)-equation cut-conservation solve in which all of the uniform stage's free alphas survive symbolically. See `derive_cut_cell_paper`'s docstring for the full rule set and `tests/fixtures/e4_1_paper_reference.json` for the anchoring fixture.
- Schemes **without** `zeros` (**E2_1, E2_2**) run the general TEMO construction in `derive_cut_cell_scheme` itself (`conserve`/non-conserve paths, the `nu=2` Neumann branch).

**History (e4-align, 2026-07):** `derive_cut_cell_mathematica` — a previously separate "singularity-free" entry point that mirrored Mathematica `explicitr-E4d1.nb` — has been deleted, along with `construct_cut_cell_with_free_gammas`, `blend_free_gammas`, `_safe_get`, `zeroed_col_default`, `zeroed_col_aza`, and the old internal `zeros`-branch inside `derive_cut_cell_scheme`. That construction was a defective one-node-downwind mis-port of the published Brady & Livescu construction (verified against the paper text; no wall row, wrong blend rule, no linear cut-conservation with surviving alphas). `derive_cut_cell_paper` is its replacement — there is no dual dispatch and no back-compat shim.

Both return a `CutCellResult` dataclass:
```python
CutCellResult(floating: Matrix, dirichlet: Matrix, neumann: Matrix|None,
              eta: list|None, dims: Dimensions, alpha_symbols: list,
              conservation_subs: dict|None=None, weights: list|None=None)
```

### Scheme catalog & dimensions
```python
SchemeParams(p, q, s, nextra, nu, zeros=())   # frozen dataclass; .dims() -> Dimensions
E2_1 = SchemeParams(p=1, q=1, s=0, nextra=1, nu=1)
E2_2 = SchemeParams(p=1, q=1, s=0, nextra=0, nu=2)
E4_1 = SchemeParams(p=2, q=3, s=0, nextra=0, nu=1, zeros=(3, 4))
E4_2 = SchemeParams(p=2, q=3, s=0, nextra=0, nu=2)   # symbolic cut-cell path is BROKEN — see gaps
compute_dimensions(p, q, s, nextra, nu) -> Dimensions(r, t, R, T, X)
```

### Uniform boundary builders
```python
derive_uniform_boundary_for_temo(scheme, alpha_symbols=None, conserve=False, zeros=None) -> UniformResult  # CURRENT general builder
build_uniform_for_mathematica(scheme, alpha_symbols=None) -> UniformResult  # alpha distribution for the mathematica path
derive_e2_uniform_boundary(scheme, ...) -> UniformResult  # LEGACY E2-only oracle — test/regression use only, NOT on the live path
```

### Building blocks
```python
# interior.py
derive_interior(s, p, nu) -> InteriorCoefficients         # exact Rational gamma/delta dicts
full_gamma_array(coeffs) -> list[Rational]                # length 2p+1, symmetry-expanded
full_delta_array(coeffs) -> list[Rational]                # length 2s+1
# boundary.py
derive_boundary(p, nu=1, s=0) -> BoundaryResult
solve_boundary_row(i, t, q, nu, free_symbols) -> BoundaryRow
# taylor_system.py
build_taylor_system(i, t, q, nu=1) -> (V, rhs)            # (q+1)×t Vandermonde + unit RHS
# conservation.py
build_conservation_system(r, t, p, boundary_rows, interior_coeffs) -> (equations, w_syms, last_row_free)
solve_conservation(equations, w_symbols, last_row_free, all_free_params, boundary_rows) -> (solution_dict, updated_rows)
# _util.py
solve_linear(A, b, unknowns) -> dict[Symbol, Expr]
```

### Codegen
```python
# codegen.py
StencilGenSpec(name, P, R, T, X, derivative_order, is_uniform, param_arrays, interior_coeffs,
               floating_coeffs, dirichlet_coeffs, has_interp=False, interp_P=0, interp_T=0,
               scalar_params=[])
generate_stencil_cpp(spec) -> str
generate_test_cpp(spec, test_cases: list[TestCase]) -> str
compute_test_values(coeffs, alpha_values, h, psi, right=False, nu=1) -> list[float]
# printer.py
StencilCodePrinter(symbol_map=None)
build_symbol_map(param_arrays, has_psi=False, scalar_params=None) -> dict[Symbol, str]
```

### Numeric PHS/RBF arm (`phs.py`)
```python
phs_stencil_weights(points, x_eval, nu, q, k=None, *, kernel="phs", epsilon=None) -> list
build_diff_matrix_rbf(n, p, q, epsilon, kernel="gaussian", nu=1, nextra=0) -> np.ndarray
build_diff_matrix_rbf_penalty(...) ; build_diff_matrix_mixed_epsilon(...)
stability_eigenvalue(n, p, q, epsilon, kernel="gaussian", nu=1, nextra=0) -> float
stability_eigenvalue_from_matrix(D) -> float
uniform_interior_weights / uniform_boundary_weights (+ _rbf / _tension variants)
```

### C++ bridge (`cpp_bridge.py`)
```python
make_brady2d_lua(scheme_type, params, *, N, t_final, template=BRADY_LIVESCU_TEMPLATE) -> str
run_cpp_brady2d(...) -> BridgeResult   # renders Lua, runs build/src/app/shoccs, parses L-inf trace
```

## How it works
The symbolic core is exact-rational throughout (SymPy `Rational`); `phs.py` is the numeric (NumPy `float`) exception. The TEMO derivation, conceptually:

1. **Interior** (`derive_interior`) — Taylor-match a `2p+1`-point circulant against derivative order `nu`, exploiting (anti)symmetry, to get exact-rational `gamma`/`delta`. (`full_gamma_array` expands to the full symmetric array.)
2. **Uniform boundary rows** (`solve_boundary_row` / `derive_uniform_boundary_for_temo`) — each row is an underdetermined Taylor system: `q+1` columns are *determined*; the remaining `t-(q+1)` are **free `alpha` constants**, carried symbolically.
3. **Discrete conservation** (`solve_conservation`) — impose telescoping/flux column-sum conditions to fix conservation (quadrature) weights `w_i` and resolve the last row's placeholder `phi_k`. The conservation equations are **bilinear** (`w_{r-1}*phi_k`); they are **linearized via theta-substitution** (`theta_k = w_{r-1}*phi_k`), solved linearly, then `phi_k = theta_k / w_{r-1}` recovered.
4. **Cut-cell rows** — extend the conserved uniform rows to embedded-boundary rows parameterized by `psi`. Schemes with a `zeros` pattern (E4_1) dispatch to `derive_cut_cell_paper`, which builds the row (wall + fluid-node-centered + wall-free transition) directly from the paper blend rule `alpha_ij = psi*alphau_ij + (1-psi)*alphau_{i-1,j-1}` and solves cut conservation linearly with the uniform stage's free alphas surviving symbolically. Schemes without `zeros` (E2_1, E2_2) use the general TEMO polynomial-ansatz construction (`construct_cut_cell_stencil`).
5. **Neumann** (`nu=2` only) — `derive_uniform_neumann` + `construct_neumann_stencil` add the Neumann variant rows and `eta` coefficients.
6. **Assembly** (`assemble_cut_cell_result`) → `CutCellResult` with `floating`/`dirichlet`/(`neumann`) matrices + conservation (quadrature) `weights`.

Codegen is a **separate, manual** step: build a `StencilGenSpec`, call `generate_stencil_cpp(spec)` (and `generate_test_cpp`), write the result to `output/`, and **hand-copy** into `src/stencils/` (occasionally hand-editing, e.g. singularity guards). See [Gotchas](#gotchas--invariants). For the full annotated flow, see `scripts/stencil_gen/docs/pipeline_reference.md`.

## How to extend
**Add a new derivative scheme:**
1. Add a `SchemeParams(p, q, s, nextra, nu, zeros=...)` constant to the catalog in `temo.py` (~line 129). Set `zeros=(...)` **only** if the scheme needs zero-constrained cut-cell conservation (like E4_1).
2. Route by `zeros`: `derive_cut_cell_scheme` dispatches internally (forwarding to `derive_cut_cell_paper` when `scheme.zeros` is non-empty), so calling it directly is always correct. `group_velocity.run_psi_sweep` (~line 750) additionally branches explicitly on `scheme.zeros` to call `derive_cut_cell_paper` directly for zeros-schemes — copy that `if scheme.zeros:` pattern if a caller needs the same short-circuit. `derive_cut_cell_paper` implements only the `nu=1` paper construction — a new `zeros`-scheme with `nu != 1` would need its own construction.
3. Emit C++: build a `StencilGenSpec` (`codegen.py:234`), call `generate_stencil_cpp(spec)` + `generate_test_cpp(spec, cases)`, write to `output/`, then **manually copy** the file into `src/stencils/` and wire it into `src/stencils/stencil.cpp`'s `from_lua` dispatch (see the [Stencils](stencils.md) reference). There is no automated codegen driver.

**Add a new RBF/spline kernel family:** add a branch in `phs._kernel_eval` / `_kernel_deriv` (~lines 179/211) and a `build_diff_matrix_*` wrapper, then thread the scalar shape parameter through `codegen.build_symbol_map(scalar_params=...)` and the C++ scheme table.

**Add a uniform-boundary distribution:** extend `derive_uniform_boundary_for_temo` (the current general builder) — **not** `derive_e2_uniform_boundary` (the retired E2-only oracle).

## Gotchas & invariants
- **`derive_cut_cell_scheme` is the single production entry point.** The former separate `derive_cut_cell_mathematica` function has been deleted (e4-align, 2026-07) — it was a defective one-node-downwind mis-port of the published Brady & Livescu (2021) construction, not a legitimate "singularity-free, current" alternative. `derive_cut_cell_scheme` now dispatches internally on `scheme.zeros`, forwarding to `derive_cut_cell_paper` (the correct paper construction) for E4_1 and running the general TEMO polynomial-ansatz construction in place for E2_1/E2_2. Nothing in the current source does psi-clamping.
- **`SYMPY_CACHE_SIZE=50000` is mandatory.** Default 1000 causes severe slowdowns on the large symbolic matrices; E4 derivation runs ~30s with it set.
- **Linearize bilinear unknowns before solving.** Any product of two unknowns (`w_{r-1}*phi_k`, cut-cell `w*gamma`) must go through theta-substitution (`solve_conservation`, `temo._solve_with_linearization`) and then recover `phi=theta/w_last`. Forgetting this makes the system nonlinear and `linsolve` fails.
- **Use `linear_eq_to_matrix` + `linsolve`, NOT `sympy.solve`** for these parameterized linear systems (project convention — performance + determinism). `_util.solve_linear` is the standard wrapper.
- **No end-to-end codegen driver.** `python -m stencil_gen generate <scheme>` prints `TODO`. `generate_stencil_cpp` is called only from tests; the `output/` → `src/stencils/` step is a manual copy and may be hand-edited afterward. `output/E4_1.cpp` already diverges from `src/stencils/E4_1.cpp` for exactly this reason.
- **Conservation is a *soft* constraint for `q>=2`.** E4_1 conservation at `nextra=0` is structurally infeasible without the `zeros` pattern — this is the documented `xfail`. Don't expect strict conservation for high-q schemes.
- **`compute_dimensions` special-cases `nu=2`.** It does NOT use the Eq.11 `nu=1` formulas; `r_eff = r-1` for `nu=2` because the last uniform boundary row overlaps the first interior row. Easy to mis-size a new `nu=2` scheme.
- **`phs.py` returns numeric (float) weights**, unlike the exact-rational symbolic core. Don't assume exact coefficients out of the PHS arm.

## Maturity & known gaps
**Verdict: mature.** Evidence: dense, real test coverage (deterministic core ~351 passed / 5 skipped per the Python ground-truth audit) plus live production callers — `group_velocity.py` dispatches to *both* cut-cell entry points (`derive_cut_cell_paper` and `derive_cut_cell_scheme`), and `output/E4_1.cpp` shares its regeneration commit with the shipped `src/stencils/E4_1.cpp`, confirming the pipeline produced real C++. The symbolic core is exact-rational and validated against hand-derived oracles; `phs.py` is exercised heavily by the sweeps/optimizer. Caveat: the core symbolic files (`temo`/`phs`/`interior`/`boundary`/`conservation`) were last touched 2026-04-06 — derivation itself is stable/frozen while the downstream analysis stack keeps evolving.

Verified dead / partial / experimental items in THIS subsystem (from audited flags):
- **`blend_free_gammas`, `construct_cut_cell_with_free_gammas`, `_safe_get`, `zeroed_col_default`, `zeroed_col_aza`, `derive_cut_cell_mathematica`** — **deleted** (e4-align, 2026-07): dead code tied to the retired mis-ported E4_1 construction. No longer present in `temo.py`.
- **`bspline_fd_weights`, `bspline_boundary_weights`, `build_diff_matrix_bspline`** (`phs.py:1009/1073/1088`) — **dead** (self-contained B-spline experiment, zero callers, no tests). Safe to delete all three. See [Cleanup Plan](../CLEANUP_PLAN.md).
- **`cut_cell_weights`** (`phs.py:965`) — **dead** (only an *unused* import in `test_phs.py:13`; no call site anywhere; thin wrapper over the tested `phs_stencil_weights`). Doc drift: `pipeline_reference.md` lists it as an entry point with a wrong signature. Safe to delete. See [Cleanup Plan](../CLEANUP_PLAN.md).
- **`derive_e2_uniform_boundary`** (`temo.py:212`) — **partial / legacy-oracle.** Fully implemented and correct but has zero production callers; retained intentionally as a regression oracle (tests pin the general builder against it). Do NOT delete — document as oracle-only.
- **`__main__.py generate/validate`** (`__main__.py:25-32`) — **experimental / never-implemented scaffold.** `generate`/`validate` print `TODO`; only `list` works (and `list` is a hardcoded string, not introspected). Real generation goes through `stencil_gen.codegen` + `output/`. Plan item 20.6a is the intended completion.
- **E4_2 symbolic cut-cell + Neumann path** (`temo.py:132` catalog + `nu==2` branches) — **partial / currently BROKEN.** `derive_cut_cell_scheme(E4_2, psi)` raises `RuntimeError: Overdetermined system inconsistent` in `build_degenerate_stencil` before reaching the Neumann step; only **E2_2** is verified through the full `nu=2` pipeline. The shipped `src/stencils/E4_2.cpp` is hand-authored and independent of this Python path. Treat the symbolic E4_2 derivation as experimental until fixed. See [Cleanup Plan](../CLEANUP_PLAN.md).
- **`conservation.py`** — the audit flagged it as "untested directly," but the verified verdict is **mature/well-covered**: `build_conservation_system`/`solve_conservation` are called directly in `conftest.py` and `test_group_velocity.py`, and the theta-recovery is validated for E4u (`n_phi=1`) and E6u (`n_phi=2`) against known-good C++ refs (~27 passing tests in `test_boundary.py`). Only the `n_phi==0` fallback branch lacks a direct test. No dedicated `test_conservation.py` exists — coverage lives in `test_boundary.py` by design.

## Tests
Run: `cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run pytest tests/ -x -q`. (~139 `@slow` tests are deselected by default.)
- **Covered:** `test_interior.py` (29), `test_boundary.py` (41, incl. conservation), `test_temo.py` (127, both entry points + uniform builders + E2_2 integration), `test_e4_cut_cell.py` (131, ~19 `@slow`), `test_codegen.py` (41), `test_codegen_e4u.py` (5), `test_printer.py` (16), `test_phs.py` (101, numeric weights + stability), `test_eval_e2_1.py` (3, builds an evaluator from `src/stencils/E2_1.cpp` to check C++/symbolic correspondence).
- **NOT covered:** no dedicated `test_conservation.py` (covered via `test_boundary.py`); the `nu=2` Neumann path is verified for **E2_2 only** — **E4_2** symbolic cut-cell is catalogued but currently raises and has no asserting test; the dead `phs` B-spline functions and `cut_cell_weights` have no tests; `__main__.py` is not directly tested.
- **Disabled/conditional:** 1 documented `xfail` (E4_1 conservation infeasibility at `nextra=0`); several `pytest.skip` guards when `sweeps/known_values.json` sub-keys are missing.

For test conventions and fixtures, see `scripts/stencil_gen/docs/testing_reference.md` and the `stencil-testing` skill.

## Related docs
- **Authoritative Python docs (do not duplicate — start here for depth):**
  - `scripts/stencil_gen/docs/pipeline_reference.md` — full annotated derivation flow, stage by stage.
  - `scripts/stencil_gen/docs/testing_reference.md` — test organization, fixtures, conventions.
  - `docs/handoff/framework_architecture.md` — high-level architecture of the whole `stencil_gen` tree.
  - `docs/handoff/known_limitations.md`, `docs/handoff/scientific_findings.md` — limitations and results.
  - Skills: `stencil-pipeline` (derivation + TEMO), `stencil-testing`, `stencil-sweeps`, `group-velocity-analysis`.
- **Downstream sweep/optimizer/stability stack** (consumers of `phs.py` + the cut-cell entry points): `scripts/stencil_gen/docs/sweeps_reference.md`, `optimization_reference.md`, `mfbo_reference.md`, `pareto_reference.md`, `group_velocity_reference.md`, `brady2d_stability_reference.md`.
- **C++ side that consumes the generated coefficients:** [Stencils](stencils.md) (the `src/stencils/` schemes + `from_lua` dispatch), [Operators](operators.md).
- **Reference workflow this mirrors:** `mathematica-files/finitedifferences/` (`taylor.wl`; `derive_cut_cell_paper`/`build_uniform_for_mathematica` follow this workflow's conventions), and `papers/BradyLivscu2021.pdf` (Brady & Livescu 2021, JCP 426:109794) for the published E4_1 cut-cell construction itself.
