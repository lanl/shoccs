# Brady-Livescu 2D Stability Cascade (`scripts/stencil_gen/stencil_gen/brady2d_stability.py`)

> **Maturity:** mature · **Audited:** 2026-05-29 · See [Capability Audit](../CAPABILITY_AUDIT.md) · [Onboarding](../ONBOARDING.md)

## Purpose
An 8-layer analytical stability-scoring cascade for finite-difference boundary closures, calibrated against the Brady & Livescu 2019 §4.3 2D varying-coefficient scalar-advection benchmark. Each layer is strictly cheaper than the next, so an unstable `(scheme, kernel, params)` candidate is rejected early (short-circuit). This is **the objective function** that drives the entire stencil parameter-optimization stack — Bayesian optimization, the staged Nelder-Mead optimizer, and the Pareto/sweep drivers all call `brady2d_stability_score`.

This doc orients and cross-links. The science (why each layer exists, calibrated thresholds, BL §4.2 derivation) is already documented in depth — do not re-derive it here. See [Related docs](#related-docs).

## Where it lives
| File | Role |
| --- | --- |
| `stencil_gen/brady2d_stability.py` | Core: all 8 layer functions, operator builders, every threshold constant, the `StabilityReport` dataclass, and the `brady2d_stability_score` orchestrator. |
| `stencil_gen/brady2d_cli.py` | CLI (argparse): single-scheme scoring, `--run-calibration`, `--alpha-basin-survey`, JSON output, `--update-known-values`. |
| `stencil_gen/brady2d/__main__.py` | One-line shim so `python -m stencil_gen.brady2d` calls `brady2d_cli.main()`. The `brady2d/` package is *only* this + an empty `__init__.py`. |
| `stencil_gen/cpp_bridge.py` | L8 Python→C++ bridge: renders the Lua template, runs the `shoccs` binary in a private tempdir, parses `logs/system.csv` for final L∞. |
| `stencil_gen/benchmarks/brady2d_calibration.py` | Runs the cascade over a fixed `FAMILIES` list (9 scheme/kernel combos), writes `known_values.json["brady2d_calibration"]`. (Module docstring is stale — see gaps.) |
| `stencil_gen/benchmarks/alpha_basin_survey.py` | Multi-seed E4 classical-α basin survey (`--alpha-basin-survey`); delegates to `optimizer.run_staged_optimize`. |
| `stencil_gen/benchmarks/brady_livescu_2d.py` | `make_coefficient_field(N)` + `L_DOMAIN`: the 2D radial flow `(c_x, c_y)` field used by L4/L5/L7. |
| `stencil_gen/benchmarks/brady_livescu_4_2.py` | `L_DOMAIN` + the §4.2 reflecting-hyperbolic model used by L3r. |

## Public API / entry points

**Central entry point** (`brady2d_stability.py`):
```python
brady2d_stability_score(scheme, kernel, params, *,
    max_layer=7, short_circuit=True,
    layer8_N=31, layer8_t_final=10.0) -> StabilityReport
```
- `scheme`: `"E2"` or `"E4"` (only these are in `_SCHEME_PARAMS`).
- `kernel`: `"classical"`, `"tension"`, `"gaussian"`, `"multiquadric"`.
- `params`: per-kernel shape — classical `{"alpha": [float, ...]}` (requires E4+; E2 classical crashes, use tension σ=0.0); tension `{"sigma": float}`; gaussian/multiquadric `{"epsilon": float}`.
- `max_layer` default **7** (both here and in the CLI); reaching L8 requires explicitly `max_layer=8`.

**`StabilityReport` dataclass** — fields: `layer1`(dict), `layer2`(`KreissResult`), `layer3`/`layer4`/`layer5`/`layer6`/`layer7`/`layer8`(dict), `layer_bl42`(dict, the L3r slot), `non_normality`(`NonNormalityReport`), `kreiss`(`KreissResult`), `overall_verdict` (`"pass"|"fail"|"unknown"`), `failed_layer`(int|None), `failed_reason`(str), `compute_time`(float). `.empty()` classmethod; `__str__` renders a per-layer table.

**Per-layer functions** (each callable standalone, all take `(scheme, kernel, params, ...)`):
`layer1_interior_boundary_gv`, `layer2_kreiss_gks`, `layer3_1d_eigenvalue`, `layer_bl42_reflecting_hyperbolic` (L3r), `layer4_local_gv_2d`, `layer5_anisotropy`, `layer6_non_normality`, `layer7_sparse_2d_eigenvalue`, `layer7_with_non_normality`, `layer8_cpp_simulation`.

**Operator builders:**
```python
build_bl42_operator(D: np.ndarray) -> scipy.sparse.csr_matrix          # L3r 2x2 block, shape (2N-2, 2N-2)
build_sparse_2d_operator(scheme, kernel, params, N) -> (L_red, keep_idx)  # L7 reduced 2D operator
```

**Module-level thresholds:** `L1_TOL=0.05`, `STABILITY_TOL=1e-10`, `L4_TOL=0.1`, `L5_TOL=0.05`, `BL42_TOL=1e-10`, `L6_TRANSIENT_GROWTH_TOL=50.0`, `L7_TOL=0.1`, `L7_TRANSIENT_GROWTH_TOL=50.0`, `L8_FINAL_LINF_TOL=1.0`. L8 dispatch table `_L8_SCHEME_TYPE`.

**CLI:** `python -m stencil_gen.brady2d` (or `... brady2d_cli`) — `--scheme {E2,E4}`, `--kernel {classical,tension,gaussian,multiquadric,phs}` (**`phs` is broken**, see gaps), `--sigma/--epsilon/--alpha/--max-layer`, plus modes `--run-calibration`, `--alpha-basin-survey`, `--json-output`, `--update-known-values`.

**L8 bridge** (`cpp_bridge.py`):
```python
run_cpp_brady2d(scheme_type, params, *, N=31, t_final=10.0, timeout=300.0,
    template=BRADY_LIVESCU_TEMPLATE, binary=SHOCCS_BINARY) -> BridgeResult
make_brady2d_lua(scheme_type, params, *, N, t_final, template)
```
`BridgeResult` fields: `final_linf`, `linf_trace`, `t_trace`, `stable`, `wall_time_s`, `exit_code`, `stderr`. `SHOCCS_BINARY = build/src/app/shoccs`.

## How it works
The 8 layers (plus the synthetic L3r slot), cheapest → most expensive:

| # | Field | What it checks | Gate |
| --- | --- | --- | --- |
| L1 | `layer1` | 1D interior+boundary group-velocity dispersion error | `boundary_gv_err > L1_TOL` |
| L2 | `layer2`/`kreiss` | GKS Kreiss determinant test | `not kr.is_stable` |
| L3 | `layer3` | 1D max Re(λ) of `-D_bc` at several N | `max_stab_eig > STABILITY_TOL` |
| **L3r** | `layer_bl42` | BL §4.2 reflecting-hyperbolic 2×2 block operator spectral abscissa | `max_spectral_abscissa > BL42_TOL` |
| L4 | `layer4` | per-point WKB local GV error on the 2D coefficient field | `max_local_gv_error > L4_TOL` |
| L5 | `layer5` | anisotropy projected on local propagation direction | `max_aligned_error > L5_TOL` |
| L6 | `layer6`/`non_normality` | 1D non-normality (spectral/numerical abscissa, Kreiss const, transient-growth bound) | `spectral_abscissa > STABILITY_TOL` or `transient_growth_bound > L6_TRANSIENT_GROWTH_TOL` |
| L7 | `layer7`/`non_normality` | sparse 2D Arnoldi spectral abscissa + 2D non-normality | `max_spectral_abscissa > L7_TOL`; combined transient-growth `> L7_TRANSIENT_GROWTH_TOL` |
| L8 | `layer8` | end-to-end `shoccs` C++ sim final L∞ | `not stable`, or `final_linf > L8_FINAL_LINF_TOL` |

Data flow: `brady2d_stability_score` runs each `if max_layer >= N:` block in order, populates the report field, calls the inner `_record_failure(layer, reason)` on a threshold breach, then `_should_stop()` (returns when `short_circuit` and a failure was recorded). Layer functions delegate to `group_velocity` (L1/L4/L5), `gks_kreiss` (L2), `phs` + classical assembler (L3/L7), `non_normality` (L6/L7), and the `benchmarks.brady_livescu_2d`/`brady_livescu_4_2` coefficient fields. L8 calls `cpp_bridge.run_cpp_brady2d` → renders `lua-configs/brady_livescu_4_3.lua` → runs `build/src/app/shoccs` in a tempdir → parses `logs/system.csv`. Calibration/sweeps persist results into `sweeps/known_values.json`.

## How to extend
- **New analytical layer:** write `layerN_*(scheme, kernel, params, ...)` returning a metrics dict; add a field to `StabilityReport`; insert an `if max_layer >= N:` block in `brady2d_stability_score` that populates it, calls `_record_failure` on breach, and checks `_should_stop()`; add a module-level `LN_TOL`; add a `__str__` branch.
- **New scheme:** add to `_SCHEME_PARAMS` (`{p, q, nextra, nu}`). Note this dict is **duplicated** in `group_velocity.py` (and conceptually in `sweeps._common`) to dodge a circular import — keep all copies in sync.
- **New kernel:** it must be understood by `group_velocity.boundary_group_velocity`, `phs.build_diff_matrix_rbf`, and `phs.stability_eigenvalue`. The non-classical branch passes `kernel` + a single `sigma`/`epsilon` scalar straight through; classical takes a separate path that requires `p >= 2` (E4+).
- **Enable a `(scheme, kernel)` for L8:** add an entry to `_L8_SCHEME_TYPE` mapping it to the Lua `scheme.type` string that `stencil::from_lua` understands, and make sure `cpp_bridge._scheme_table_for` emits the right params.
- **New calibration family:** append to `FAMILIES` in `benchmarks/brady2d_calibration.py`.

## Gotchas & invariants
- **L3r runs inside the L3 tier** (`if max_layer >= 3`). Requesting `max_layer=3` evaluates *both* L3 and L3r and can short-circuit-fail at L3r. This is why E4 tension σ=3.0 (which passes L3) reports overall FAIL at "layer 3" — `failed_layer=3` covers both.
- **Layer index vs name mismatch.** L3r lives in `StabilityReport.layer_bl42` and records `failed_layer=3`, but BO/optimizer/pareto address it via the synthetic **fidelity index 5** and the dotted alias `layer_bl42.max_spectral_abscissa`. This is intentional and consistent (index 5 lets the BO ICM kernel separate L3 from L3r; the populating-layer alias is 3 via `optimizer._FIELD_LAYER_ALIAS`), but it is an easy double-take. Do not assume layer numbers are contiguous.
- **Grid-spacing rescale.** `build_diff_matrix_rbf` and the classical assembler return weights for **unit spacing** (`h=1`). `build_sparse_2d_operator` and `build_bl42_operator` manually divide `D` by `h = L_DOMAIN/(N-1)`. Forgetting this silently shifts the spectral abscissa and miscalibrates the L7/L3r thresholds (the `L7_TOL=0.1` calibration assumes correct `h` scaling).
- **2D flattening is Fortran/column-major:** `u_flat[j*N+i] = u[i,j]`, `Dx_2D = kron(I, D1)`, `Dy_2D = kron(D1, I)`, coefficient fields flattened with `.flatten("F")`. Mixing C-order corrupts the operator.
- **L7 spectral abscissa is intentionally positive.** The BL radial field has `div(c) = 1/psi > 0`, so the continuous operator is non-skew-symmetric. `L7_TOL=0.1` separates known-stable (~0.018) from known-unstable (~3.1) — it is *not* a "must be ≤ 0" test like L3.
- **E2 + classical crashes** (negative symbol count in `derive_boundary` at `p=1`). Use `kernel="tension"`, `sigma=0.0` for E2 — that dispatches to PHS k=2 inside `phs.py`. This is the path the calibration uses for the `E2_phs_k2`/`E4_phs_k2` families.
- **L8 isolation/semantics.** Each L8 call uses a private `TemporaryDirectory`, so parallel runs are safe. The bridge's `stable` flag (finite L∞ below the hard blow-up ceiling) is distinct from the `L8_FINAL_LINF_TOL=1.0` accuracy gate applied in the orchestrator.

## Maturity & known gaps
**Mature.** All 8 layers are fully implemented (no stubs; L8 raises `NotImplementedError` only for unregistered `(scheme, kernel)` pairs, by design). It is a live central dependency of `bo.py`, `optimizer.py`, `pareto.py`, `sweeps/*`, and `tools/multimodal_sweep.py`, with stored regression data in `known_values.json`. The test suite is 1527 lines / ~105 tests; the audit re-ran the non-slow subset: **93 passed, 12 deselected (slow), 0 failures**. The analytical core L1–L7 is solid; a few rough edges remain.

Verified flags for this subsystem (from the audit JSON):
- **CLI `--kernel phs` is broken end-to-end** (`partial`). The CLI advertises and validates `phs` (`brady2d_cli.py:135`/`:240`) but no layer function has a `phs` branch — it forwards to `phs_stencil_weights` without the required `k`, raising `ValueError("k is required for PHS kernel")` even at `max_layer=1`. PHS is actually reachable today via `kernel="tension"`, `sigma=0.0` (→ PHS k=2). Recommendation: finish (add `--phs-order/-k` and thread it) or remove the choice. See [Cleanup Plan](../CLEANUP_PLAN.md).
- **L8 real-binary path requires a built `shoccs` binary.** The Python layer is fully implemented and heavily unit-tested via monkeypatch, and is wired into three sweep CLIs. The C++ build is green (fixed 2026-06-04; was the Kokkos 5.1 `create_graph` break) and `build/src/app/shoccs` now builds and loads against Kokkos 5.1.1, so a real L8 run executes once the C++ tree is built (`cmake --build build`). Caveat: the `.exists()` guards check only file presence, so if the binary is absent the slow real-binary tests/sweeps will *run and fail*, not skip cleanly. L8 also supports **E4 only** (`_L8_SCHEME_TYPE` has no E2 entries; E2 deferred to plan 42.10a). See [Cleanup Plan](../CLEANUP_PLAN.md).
- **`_build_classical_diff_matrix` E2 unusability** — investigated and **refuted as a gap** (`mature`). The E2-classical limitation is a documented, deliberately guarded design boundary (E2 has no free α params); no caller can reach it via `FAMILIES` or `DEFAULT_BOUNDS`, and manual misuse fails loudly. Keep as-is.
- **`alpha_basin_survey.py` "no tests / unvalidated cut-cell flag"** — investigated and **refuted** (`mature`). `TestAlphaBasinSurvey` (5 passing tests) lives in `test_optimizer.py`; the `cpp_cutcell_violates_197_288` flag is a fully-computed, tested diagnostic in `optimizer.py` that the survey merely propagates.
- **L3r index-5 "collision"** — investigated and **refuted** (`mature`). Intentional two-level naming (external fidelity index 5 vs populating-layer 3), resolved through a single shared `_FIELD_LAYER_ALIAS`/`_infer_max_layer` source of truth, covered by a dedicated end-to-end BO CLI test.

**Stale docstrings (not behavior bugs, but will mislead):**
- `brady2d_stability.py` module docstring lists only L1–L7 (omits L8); line 1009 (`layer7_with_non_normality`) says `L7_TOL (5e-3)` but the real value is `0.1`; line 98 calls `layer6` "reserved" though it is actively populated; `brady2d_cli.py` `--max-layer` help says "(1-7)" though 8 is valid.
- `benchmarks/brady2d_calibration.py` module docstring (dated 2026-04-12) claims "all E4 families … pass through L6" — **wrong** since L3r was wired in (plan 44.4). Per `known_values.json`, only `E4_classical` and `E2_phs_k2` pass; `E4_phs_k2`/`E4_tension_3`/`E4_gaussian_09`/`E4_multiquadric_1` now fail at L3r, and the three E2 families still fail at L1.

## Tests
- `tests/test_brady2d_stability.py` (1527 lines, ~105 tests) — a dedicated `TestLayerN` class per layer plus `TestLayerBL42`, `TestLayer7WithNonNormality`, `TestLayer8`/`TestLayer8Dispatch`/`TestStabilityScoreL8`, `TestBuildBL42Operator`, `TestBuildSparse2D`, `TestStabilityScoreOrchestrator`, `TestBrady2DScoreL3rCascade`, `TestStabilityReport`. 93 non-slow pass (~92s); ~12 `@pytest.mark.slow` gate the 2D Arnoldi (L7), the combined L7+non-normality check, and real-binary L8.
- `tests/test_cpp_bridge.py` (31 tests) covers the bridge.
- `TestAlphaBasinSurvey` (5 tests) lives in `tests/test_optimizer.py`, not a separate file.
- **Not covered:** `brady2d_cli.py` argument parsing/validation has no `test_brady2d_cli.py` — which is exactly why the broken `--kernel phs` path went uncaught. No layer is exercised with `kernel="phs"`.
- **Conditional skips:** L8 unit tests monkeypatch `run_cpp_brady2d`. Several tests `pytest.skip` when a key is absent from `known_values.json`. Note the real-binary slow tests guard on `SHOCCS_BINARY.exists()` only, so if the binary is missing they fail rather than skip; build the C++ tree first (`cmake --build build`) to run them.

Run from `scripts/stencil_gen`: `SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_brady2d_stability.py -q` (add `-m "not slow"` to skip the expensive layers).

## Related docs
- [`scripts/stencil_gen/docs/brady2d_stability_reference.md`](../../scripts/stencil_gen/docs/brady2d_stability_reference.md) — full per-layer reference, the §4.3 problem statement, and the objective/fidelity-field naming used by the sweep/BO/Pareto drivers. **Primary deep reference.**
- [`scripts/stencil_gen/docs/bl42_reference.md`](../../scripts/stencil_gen/docs/bl42_reference.md) — the L3r / BL §4.2 reflecting-hyperbolic model derivation.
- [`docs/handoff/scientific_findings.md`](../../docs/handoff/scientific_findings.md) — non-obvious results (e.g. tension splines are universally L3r-infeasible) that drive optimization design.
- Optimization consumers: [`scripts/stencil_gen/docs/optimization_reference.md`](../../scripts/stencil_gen/docs/optimization_reference.md), [`mfbo_reference.md`](../../scripts/stencil_gen/docs/mfbo_reference.md), [`pareto_reference.md`](../../scripts/stencil_gen/docs/pareto_reference.md).
- Other reference docs: see the [Capability Audit](../CAPABILITY_AUDIT.md) index for `py-group-velocity`, `py-phs`, `py-optimizer`, and the C++ `stencils`/`systems` subsystems consumed by L8.
