# Python→C++ Stencil Bridge (`scripts/stencil_gen/stencil_gen/cpp_bridge.py`)

> **Maturity:** mature · **Audited:** 2026-05-29 · See [Capability Audit](../CAPABILITY_AUDIT.md) · [Onboarding](../ONBOARDING.md)

## Purpose
This subsystem closes the loop between the SymPy stencil-optimization stack (the `stencil_gen` Python package) and the compiled `shoccs` C++ solver. It renders a Lua config from a template with runtime-parameterized boundary-closure stencil parameters (`alpha`/`sigma`/`epsilon`), invokes the prebuilt `shoccs` binary on the Brady-Livescu 2019 §4.3 2D varying-coefficient scalar-wave test, parses `logs/system.csv`, and returns a `BridgeResult` (final L∞, stability verdict, traces, exit code). It is **Layer 8 (L8)** of the brady2d analytical stability pipeline: the empirical end-to-end validator that confirms analytical predictions (L1–L7) survive contact with the real solver. The key trick is that the stencil parameter is passed through Lua at runtime and read in the C++ stencil constructor — a single compiled binary serves every point of a parameter sweep with **no per-point recompile**.

## Where it lives
| File | Role |
| --- | --- |
| `scripts/stencil_gen/stencil_gen/cpp_bridge.py` | The bridge itself: `make_brady2d_lua`, `run_cpp_brady2d`, `BridgeResult`, path constants |
| `scripts/stencil_gen/stencil_gen/brady2d_stability.py` | L8 wrappers `layer8_cpp_simulation` (line 1059) and `brady2d_stability_score` (line 1140); the `_L8_SCHEME_TYPE` dispatch table (line 1051) |
| `lua-configs/brady_livescu_4_3.lua` | The template with `--{{N}}--` / `--{{T_FINAL}}--` / `--{{SCHEME_TABLE}}--` markers; **not** standalone-runnable |
| `lua-configs/brady_livescu_4_3_n61.lua`, `_long.lua` | Standalone (non-templated) variants for manual N=61 and t=100 runs |
| `src/stencils/stencil.cpp` | `stencil::from_lua` dispatch (lines 44–78) reads `sigma`/`epsilon` and calls `make_*_E4u_1` — where the runtime param enters C++ |
| `src/stencils/tension_E4u_1.cpp` | Representative runtime-parameterized family (siblings `gaussian_E4u_1.cpp`, `multiquadric_E4u_1.cpp`); ctor solves the RBF+poly system at construction and caches coeffs |
| `scripts/stencil_gen/sweeps/{brady2d_sweep,optimize,bo}.py` | Three sweep drivers consuming the bridge via `--validate-with-cpp` (`bo.py` is the plan-47 MF-BO driver) |
| `docs/brady2d_cpp_bridge_reference.md` | Existing deep-dive reference doc (see Related docs; mostly accurate, a few stale defaults noted below) |

## Public API / entry points

**Python bridge (`cpp_bridge.py`):**
```python
@dataclass
class BridgeResult:
    final_linf: float = float("nan")
    linf_trace: np.ndarray = ...   # empty by default
    t_trace:    np.ndarray = ...   # empty by default
    stable: bool = False
    wall_time_s: float = 0.0
    exit_code: int = 0
    stderr: str = ""

def run_cpp_brady2d(scheme_type, params, *, N=31, t_final=10.0, timeout=300.0,
                    template=BRADY_LIVESCU_TEMPLATE, binary=SHOCCS_BINARY) -> BridgeResult
def make_brady2d_lua(scheme_type, params, *, N, t_final,
                     template=BRADY_LIVESCU_TEMPLATE) -> str

# module constants
REPO_ROOT, LUA_TEMPLATE_DIR, BRADY_LIVESCU_TEMPLATE, SHOCCS_BINARY  # = build/src/app/shoccs
```

**Upstream L8 wrappers (`brady2d_stability.py`):**
```python
def layer8_cpp_simulation(scheme, kernel, params, *, N=31, t_final=10.0) -> dict
    # -> {final_linf, stable, wall_time_s, bridge_result}
def brady2d_stability_score(scheme, kernel, params, *, max_layer=7,
                            short_circuit=True, layer8_N=31, layer8_t_final=10.0) -> StabilityReport
    # runs L8 only when max_layer >= 8
```

**C++ side (`src/stencils/`):** `stencil::from_lua(const sol::table&, const logs&={})` dispatches on `type` (order==1 branch) to:
- `make_tension_E4u_1(real sigma)` — reads `m["sigma"].get_or(3.0)`
- `make_gaussian_E4u_1(real epsilon)` — reads `m["epsilon"].get_or(0.9)`
- `make_multiquadric_E4u_1(real epsilon)` — reads `m["epsilon"].get_or(1.0)`
- `make_E4u_1(alpha)` / `make_E4_1` / `make_E2_1` / `make_E6u_1` / `make_E8u_1` (classical, `alpha` array)

(declared in `src/stencils/stencil.hpp:277–284`)

## How it works
The full data flow for a single L8 validation call:

1. A sweep driver calls `brady2d_stability_score(scheme, kernel, params, max_layer=8)`.
2. When `max_layer >= 8`, it calls `layer8_cpp_simulation(scheme, kernel, params, ...)`.
3. That looks up `(scheme, kernel)` in `_L8_SCHEME_TYPE` → a Lua `scheme.type` string (e.g. `("E4","tension") → "tension_E4u"`), raising `NotImplementedError` for unsupported combos.
4. It calls `run_cpp_brady2d(scheme_type, params, N, t_final)`, which:
   - calls `make_brady2d_lua(...)` to `str.replace` the three markers (`--{{N}}--`, `--{{T_FINAL}}--`, `--{{SCHEME_TABLE}}--`) — `_scheme_table_for` emits whichever of `alpha`/`sigma`/`epsilon` are present in `params`;
   - creates a per-call `TemporaryDirectory`, writes the Lua there plus an empty `logs/` dir, and runs `shoccs <lua>` with `cwd=tmpdir`;
   - on the C++ side, `simulation_cycle::from_lua` parses the Lua and `stencil::from_lua` reads `sigma`/`epsilon` and calls the matching `make_*_E4u_1` constructor (the RBF+poly solve happens once, at construction);
   - the scalar-wave system writes `logs/system.csv`; `_parse_system_csv` reads **column 1 = Time, column 3 = L∞** from rows whose first cell starts with a digit;
   - returns a `BridgeResult` with the final L∞ and a `stable` flag.

The runtime-parameterization invariant is the whole point: the scalar param travels `params dict → Lua scheme table → C++ constructor`, so one compiled binary serves an entire parameter sweep.

## How to extend
**Add a new runtime-parameterized boundary-closure family (end to end):**
1. **C++:** add `src/stencils/<name>_E4u_1.cpp` implementing the construction-time RBF+poly solve (copy `tension_E4u_1.cpp`); declare `make_<name>_E4u_1(real param)` in `stencil.hpp` (near line 282); add an `else if (type == "<name>_E4u")` branch in `stencil::from_lua` (`stencil.cpp:60–77`) reading `m["sigma"|"epsilon"].get_or(default)`; add a Catch2 test + fixture (model on `t-tension_E4u_1`).
2. **Rebuild `shoccs` once.** (See [app-and-build](app-and-build.md).)
3. **Python (only if a new scalar name is needed):** `_scheme_table_for` (`cpp_bridge.py:52`) already emits any of `alpha`/`sigma`/`epsilon` present in `params`; add a new key there only for a differently-named scalar.
4. **Wire dispatch:** add `(scheme, kernel) → "<name>_E4u"` to `_L8_SCHEME_TYPE` (`brady2d_stability.py:1051`).
5. The three sweep CLIs pick it up automatically via `brady2d_stability_score(max_layer=8)`.

For codegen-emitted families, `StencilGenSpec.scalar_params` + `printer.build_symbol_map` already generalize the `alpha` runtime-param pattern; plan 42's three spline families are deliberately hand-written, not codegen-emitted (see [py-derivation](py-derivation.md)).

To change failure thresholds: the bridge's hard blow-up ceiling is `final_linf < 10.0` in `run_cpp_brady2d` (`cpp_bridge.py:195`); the sweep-layer pass/fail tolerance `L8_FINAL_LINF_TOL = 1.0` lives separately (`brady2d_stability.py:1045`).

## Gotchas & invariants
- **Two distinct stability thresholds — easy to confuse.** `BridgeResult.stable` uses `final_linf < 10.0` (a loose blow-up ceiling, `cpp_bridge.py:195`); the sweep/stability layer uses `L8_FINAL_LINF_TOL = 1.0`. A run can be `stable=True` yet fail the sweep's L8 tolerance.
- **CSV parse is positional and brittle.** `_parse_system_csv` hardcodes Time=column 1, L∞=column 3, treating any row whose first cell starts with a digit as data. If `scalar_wave.cpp` reorders/renames `system.csv` columns, the bridge silently parses wrong values rather than erroring. This is the tightest coupling point to the C++ output format.
- **`make_brady2d_lua` does pure `str.replace` with no validation.** A missing/typo'd marker silently yields broken Lua that fails only when `shoccs` parses it (surfacing as a nonzero exit, not a clear Python error). Brace balance is checked in tests, not at runtime.
- **Lua numbers are emitted via `repr(float(x))` deliberately** so `alpha` coefficients round-trip to full double precision matching `known_values.json`. Do **not** "simplify" to `str()` or f-string formatting — it would truncate precision.
- **L8 is E4-uniform-only.** `_L8_SCHEME_TYPE` has only the four E4 combos; `scheme="E2"` raises `NotImplementedError` (deferred, plan 42.10a). Cut-cell (non-uniform-ψ) runtime parameterization is also deferred (plan 42.10b) — the spline families solve once because Brady-Livescu §4.3 is a uniform rectangular domain.
- **L8 is diagnostic-only.** `optimize.py` / `bo.py` do **not** alter `best_objective` when L8 disagrees with the analytical verdict (plan 43.10a). A failing C++ run is a flag to investigate, not an automatic rejection.
- **Concurrency safety depends entirely on the per-call `TemporaryDirectory` cwd.** `shoccs` writes `logs/system.csv` under its cwd, so private tempdirs isolate parallel sweep runs. If a caller ever passes a fixed cwd or the binary writes logs to an absolute path, parallel runs would race on `system.csv`.
- **`REPO_ROOT = Path(__file__).resolve().parents[3]`.** Moving `cpp_bridge.py` to a different directory depth silently breaks every path constant (`BRADY_LIVESCU_TEMPLATE`, `SHOCCS_BINARY`).
- **The template is not standalone-runnable** — its markers are Lua line comments (`--{{...}}--`). Use `brady_livescu_4_3_n61.lua` / `_long.lua` for manual runs.

## Maturity & known gaps
**Verdict: mature.** Evidence: `cpp_bridge.py` is complete (all error paths handled: nonzero exit, timeout, missing CSV, empty CSV) with strong hermetic Python coverage; the three C++ spline families are real, registered in `from_lua`, and each has a passing Catch2 test; and real callers route through the bridge across three sweep drivers (`brady2d_sweep`, `optimize`, `bo` — the latter is the active plan-47 MF-BO driver). The bridge landed in plan 42 (Apr 2026) and `bo.py` shows continued active use in plan 47 (May 2026, the current plan).

Known gaps (from verified audit flags):
- **Real-binary end-to-end smoke test requires a built binary.** `TestCppBridgeSmoke.test_classical_e4u_short_run` (`tests/test_cpp_bridge.py:329`) is `@pytest.mark.slow` (opt-in via `--run-slow`) and skips when `SHOCCS_BINARY` is absent. The binary builds and loads now (build fixed 2026-06-04 — was the Kokkos 5.1 `create_graph` break, since resolved), so `build/src/app/shoccs` runs and the Python↔compiled-shoccs handshake can be exercised end to end. The test still needs the binary built and is slow, so it stays opt-in. *Status: working; see [app-and-build](app-and-build.md) and [Cleanup Plan](../CLEANUP_PLAN.md). Equivalent slow/binary-guarded E2E tests live at `test_brady2d_stability.py:1212/1445/1501` and `test_optimizer.py:2449`.*
- **E2 + cut-cell L8 dispatch raises `NotImplementedError` by design.** `_L8_SCHEME_TYPE` mirrors the C++ binary's real capability set (E4-uniform only); the binary genuinely cannot run E2-spline or spline-cut-cell closures (no such source files exist). *Status: partial-by-design; deferred to plans 42.10a/42.10b (both still open). Not a stub — the guard is unit-tested. Adding support is a feature add (land the C++ Lua scheme types first, then add table rows), not a fix.*
- **`lua-configs/brady_livescu_4_3_long.lua` (and `_n61.lua`) have no automated consumer.** Valid, end-to-end-runnable standalone configs for manual long-time/N=61 stability checks, documented in plan 42 and the handoff docs — but no pytest/ctest exercises them, so a future schema change could silently break them. *Status: partial (keep; deliberate manual-use artifact, not dead code).*
- **`_run_cpp_validation` is duplicated in `optimize.py` and `bo.py`.** Both wrappers are complete, CLI-wired, and tested (not partial) — but the shared constants `_CPP_SUPPORTED_KERNELS`/`_SCHEMES`/`_CPP_VALIDATION_N_DEFAULT`/`_T_FINAL_DEFAULT` are re-declared verbatim in both (and `pareto.py`), a genuine single-source-of-truth hazard. *Status: mature code, cleanup candidate — hoist shared constants into a common module; see [Cleanup Plan](../CLEANUP_PLAN.md).*

## Tests
- **`scripts/stencil_gen/tests/test_cpp_bridge.py`** (31 tests, hermetic via a fake `shoccs` shell script): `TestMakeBrady2DLua` / `TestMakeBrady2DLuaSpline` (marker substitution, alpha/sigma/epsilon scheme-table emission, brace balance, `repr()` precision); `TestRunCppBrady2D` (success parse, nonzero exit, unstable/large-L∞, missing CSV, empty CSV, graceful timeout, tempdir isolation); `TestBridgeResultDefaults`. The full driver logic is exercised against a fake binary, so the real binary is not needed for these.
- **`scripts/stencil_gen/tests/test_brady2d_stability.py`** — L8 dispatch tested by monkeypatching `run_cpp_brady2d` to a stub (classical→E4u, spline kernel dispatch, `NotImplementedError` on unknown `(scheme,kernel)`, default N/t_final forwarding, cascade with `max_layer=8`).
- **`scripts/stencil_gen/tests/{test_optimizer.py,test_sweep_bo.py}`** — `TestRunCppValidation` / `TestValidateWithCpp` cover the two sweep wrappers (`test_sweep_bo.py -k ValidateWithCpp` → 13 passed).
- **C++ Catch2:** `t-tension_E4u_1`, `t-gaussian_E4u_1`, `t-multiquadric_E4u_1` (label `stencils`) assert coefficient match to `phs._rbf_weights_numeric` within 1e-12.
- **NOT covered by default:** the actual Python→compiled-`shoccs` handshake (CSV column layout, exit semantics, Lua parse of the rendered template). The only true E2E tests are `@pytest.mark.slow` + binary-guarded, so they run only with a built binary and `--run-slow` (the binary builds and loads as of the 2026-06-04 build fix; see Maturity & known gaps). No test asserts the rendered Lua actually parses in C++ `sol`/Lua.

## Related docs
- [`docs/brady2d_cpp_bridge_reference.md`](../brady2d_cpp_bridge_reference.md) — the full deep-dive reference for this bridge (data flow, examples, failure tables). Mostly accurate; note its L8 defaults (N=31/t_final=10.0) match the `layer8_cpp_simulation` *signatures* but not the actual CLI invocations (`brady2d_sweep` uses `layer8_N=21`, `layer8_t_final=1.0`; `optimize.py` has its own `_CPP_VALIDATION_*` defaults), and its failure table conflates `BridgeResult.stable` (< 10.0) with the sweep-layer `L8_FINAL_LINF_TOL` (1.0).
- [`scripts/stencil_gen/docs/brady2d_stability_reference.md`](../../scripts/stencil_gen/docs/brady2d_stability_reference.md) — the L1–L8 analytical stability stack this bridge plugs into as L8.
- [py-brady2d](py-brady2d.md) — the analytical brady2d stability stack (L1–L7) and sweep drivers (Python side).
- [py-derivation](py-derivation.md) — the SymPy stencil-derivation/codegen pipeline that produces the families this bridge validates.
- [stencils](stencils.md) — the C++ stencil subsystem and `stencil::from_lua` (the dispatch this bridge targets).
- [systems](systems.md) — `scalar_wave`, the C++ system that writes `logs/system.csv` (the bridge's parse contract depends on its column layout).
- [app-and-build](app-and-build.md) — the `shoccs` binary build (green as of 2026-06-04; real-binary L8 runs work again).
- `plans/42-cpp-bridge-runtime-parameterized-stencils.md` — the plan that introduced this bridge (pre-Kokkos-migration rationale archive lives under `plans/meta.md`).
