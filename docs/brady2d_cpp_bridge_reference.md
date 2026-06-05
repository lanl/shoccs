# Brady-Livescu 2D C++ Bridge Reference

Reference for the Python → C++ closed-loop bridge added in plan 42. The bridge
lets plan 41's analytical stability stack (layers L1–L7) validate survivors by
running the compiled `shoccs` binary on the Brady-Livescu §4.3 two-dimensional
varying-coefficient scalar wave test as a final layer (L8), without rebuilding
C++ per sweep point.

## Architecture

```
Python sweep
  → brady2d_stability_score(max_layer=8)
    → layer8_cpp_simulation(scheme, kernel, params)
      → run_cpp_brady2d()
        → make_brady2d_lua()        renders Lua config from template
        → subprocess.run(shoccs …)  in a per-call tempfile.TemporaryDirectory
        → parses logs/system.csv    returns BridgeResult
  ← StabilityReport with layer8 = {final_linf, stable, wall_time_s, bridge_result}
```

- **No rebuild per sweep point.** The boundary-closure parameter (`alpha`,
  `sigma`, or `epsilon`) is passed through Lua at runtime; the stencil struct
  reads it in its constructor. A single compiled `shoccs` binary serves every
  point of a sweep.
- **Concurrency-safe.** Each `run_cpp_brady2d` call uses its own
  `tempfile.TemporaryDirectory` as `cwd`, so concurrent invocations write
  `logs/system.csv` under disjoint roots and never race on
  `REPO_ROOT/logs/`.
- **Isolation boundary.** Python owns optimization and parameter-space
  sweeping; C++ is a validator. Python never links the solver, C++ never
  reads `known_values.json`.

### Key files

| File | Role |
|------|------|
| `lua-configs/brady_livescu_4_3.lua` | Template with `--{{N}}--`, `--{{T_FINAL}}--`, `--{{SCHEME_TABLE}}--` markers (not standalone-runnable) |
| `lua-configs/brady_livescu_4_3_n61.lua`, `…_long.lua` | Thin standalone variants (N=61, or N=31 at t=100) |
| `scripts/stencil_gen/stencil_gen/cpp_bridge.py` | `make_brady2d_lua`, `run_cpp_brady2d`, `BridgeResult` |
| `scripts/stencil_gen/stencil_gen/brady2d_stability.py` | `layer8_cpp_simulation`, `brady2d_stability_score(max_layer=8)` |
| `scripts/stencil_gen/sweeps/brady2d_sweep.py` | `brady2d` subcommand for parameter sweeps (`--validate-with-cpp` re-runs top-K at L8) |

## C++ stencil families added in plan 42

Each family is a uniform (non-cut-cell) first-derivative 4th-order boundary
closure keyed on a single scalar parameter. All three live under
`src/stencils/` and are registered in `stencil::from_lua`.

| Family | Lua `scheme.type` | Parameter (Lua key, C++ struct field) | Default |
|--------|-------------------|---------------------------------------|---------|
| `tension_E4u_1` | `"tension_E4u"` | `sigma` (`real sigma`) | 3.0 |
| `gaussian_E4u_1` | `"gaussian_E4u"` | `epsilon` (`real epsilon`) | 0.9 |
| `multiquadric_E4u_1` | `"multiquadric_E4u"` | `epsilon` (`real epsilon`) | 1.0 |

Construction-time solve. Each constructor takes the scalar parameter,
builds the 10×10 augmented RBF+polynomial system (6-point tension/Gaussian/
multiquadric kernel plus the four monomials `1, x, x^2, x^3` for `q=3`),
runs one hand-coded Gaussian elimination with partial pivoting per boundary
row, and caches the 5×7 = 35-entry coefficient block in `cached_coeffs`.
`nbs_floating` then copies from the cache and applies the `1/h` scaling plus
the `right=true` negate+reverse flip at each call site. Layout:

- Rows 0..3 × cols 0..5: solved RBF+poly weights at `h=1`.
- Rows 0..3 × col 6: zero-padded.
- Row 4: hardcoded classical E4 centered stencil
  `{0, 0, 1/12, -2/3, 0, 2/3, -1/12}`.

The Python reference is `phs._rbf_weights_numeric`; each family ships with a
fixture under `scripts/stencil_gen/tests/fixtures/` and a Catch2 test
(`t-tension_E4u_1`, `t-gaussian_E4u_1`, `t-multiquadric_E4u_1`) that asserts
match within `1e-12`.

## Runtime-parameterized codegen pattern

`StencilGenSpec` (`scripts/stencil_gen/stencil_gen/codegen.py`) accepts scalar
runtime parameters via the `scalar_params: list[str]` field — companion to
the existing `param_arrays`.

```python
spec = StencilGenSpec(
    name="TensionE4u1",
    scalar_params=["sigma"],  # → `real sigma;` field + `TensionE4u1(real sigma_)` ctor
    ...
)
```

The `StencilCodePrinter` (`printer.py::build_symbol_map`) maps
`Symbol("sigma") → "sigma"` with no subscript, so SymPy expressions containing
`Symbol("sigma")` print as `sigma` (not `sigma[0]`). This lets the pattern
already used by `E4_1` for `alpha` generalize to scalar parameters without
retouching the printer per family.

Plan 42's three spline families are hand-written rather than codegen-emitted,
because emitting a correct Gaussian elimination from SymPy adds significant
scope. Scalar-param codegen is in place for future families that can be
expressed in closed form (see plan 42 42.4).

## Why cut-cell variants are deferred

Brady-Livescu §4.3 is posed on a uniform rectangular domain, so the boundary
closure does not depend on `psi`. The construction-time solve therefore runs
exactly once per simulation and the coefficients are cached.

Cut-cell (non-uniform-`psi`) runtime parameterization needs either (a) a
per-cut-cell runtime solve, or (b) a precomputed coefficient cache indexed by
`psi`. Both are strictly larger scope than plan 42 and are tracked as plan
42.10b.

## Bridge usage

### Programmatic

```python
from stencil_gen.cpp_bridge import run_cpp_brady2d

res = run_cpp_brady2d(
    scheme_type="tension_E4u",
    params={"sigma": 3.0},
    N=31,
    t_final=10.0,
)
# res.final_linf, res.stable, res.wall_time_s, res.exit_code, res.stderr
```

### Full-pipeline with L8

```python
from stencil_gen.brady2d_stability import brady2d_stability_score

rep = brady2d_stability_score(
    scheme="E4",
    kernel="tension",
    params={"sigma": 3.0},
    max_layer=8,
    layer8_N=31,
    layer8_t_final=10.0,
)
# rep.overall_verdict, rep.layer8["stable"], rep.layer8["final_linf"]
```

### Sweep with empirical validation

```bash
cd scripts/stencil_gen
uv run python -m sweeps brady2d \
    --scheme E4 --kernel tension \
    --param-range 2 6 20 \
    --max-layer 3 \
    --validate-with-cpp
```

Top 3 passing points (ranked by `layer6.transient_growth_bound` when present,
else `layer3.max_stab_eig`) are re-run at `max_layer=8`.

## Failure thresholds

| Layer | Metric | Constant | Value |
|-------|--------|----------|-------|
| L8 | `final_linf` at `t_final` | `L8_FINAL_LINF_TOL` | 1.0 |
| L8 | `stable` | parse of `logs/system.csv` last row | `isfinite(final_linf) and final_linf < 10.0` |

`BridgeResult.final_linf = NaN` and `stable = False` on nonzero exit, timeout,
or missing/empty CSV; the diagnostic is in `stderr` and `exit_code`.
