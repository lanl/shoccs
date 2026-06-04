# Stencils (`src/stencils/`)

> **Maturity:** partial · **Audited:** 2026-05-29 · See [Capability Audit](../CAPABILITY_AUDIT.md) · [Onboarding](../ONBOARDING.md)

## Purpose
The stencils subsystem produces the finite-difference *coefficients* the solver discretizes derivatives with. Each "scheme" bundles an **interior circulant operator** (a centered stencil applied on the regular grid) with a **dense boundary closure** — the numerical boundary scheme, abbreviated `nbs` — and, for cut-cell schemes, **wall interpolation** coefficients. A single type-erased `stencil` handle wraps any concrete scheme and is the only object the operators subsystem (derivative/gradient/laplacian) ever consumes. `stencil::from_lua` maps a Lua `scheme = {order, type, ...}` table to a concrete scheme at builder time, which is how production simulations select a discretization.

## Where it lives
| File | Role |
| --- | --- |
| `src/stencils/stencil.hpp` | Public header: the `Stencil` concept, `info`/`interp_info`/`interp_line` structs, the type-erased `stencil` class (including the `interp()` cut-cell dispatch), all `make_*` factory declarations, the `second::E2`/`second::E4` externs, and the `ccs::stencil` alias. |
| `src/stencils/stencil.cpp` | `stencil::from_lua` — **THE** production dispatch mapping `(order, type)` + `alpha`/`sigma`/`epsilon` to a concrete scheme. Only `order==2` and `order==1` branches exist. |
| `src/stencils/E2_1.cpp` | 1st-derivative E2 cut-cell scheme. Full `interp_interior`/`interp_wall` (`query_interp() == {2,3}`). `alpha[4]`. |
| `src/stencils/E4_1.cpp` | 1st-derivative E4 **cut-cell** scheme (psi-dependent coefficients). Production copy with hand-added singularity guards (see Gotchas). `query_interp() == {}`. `alpha[2]`. |
| `src/stencils/E2_2.cpp` | 2nd-derivative E2 (Laplacian) scheme; defines the `second::E2` global. Supports cut-cell interp and Neumann (`nextra=2`). Used by `heat` (`order=2, type=E2`). |
| `src/stencils/E4_2.cpp` | 2nd-derivative E4 scheme; defines `second::E4`. Cut-cell interp + Neumann. Reachable via `order=2, type=E4` but config-orphaned. |
| `src/stencils/E4u_1.cpp` | **Uniform-mesh** E4 1st-derivative boundary closure. `alpha[2]`; ignores psi; no interp. The only first-derivative scheme used in a real config (`brady_livescu`). |
| `src/stencils/E6u_1.cpp` | Uniform E6 1st-derivative closure (`P=3,R=5,T=8`), `alpha[5]`. Wired + tested, no production config. |
| `src/stencils/E8u_1.cpp` | Uniform E8 1st-derivative closure (`P=4,R=7,T=11`), `alpha[7]`. Wired + tested, no production config. |
| `src/stencils/polyE2_1.cpp` | Polynomial-augmented E2 1st-derivative scheme with separate floating/dirichlet/interpolant alpha arrays; full cut-cell interp (`query_interp() == {2,4}`). Used by `scalar_wave.lua` & `eigenvalues.lua` (`type=E2-poly`). |
| `src/stencils/tension_E4u_1.cpp` | **Experimental** RBF (tension-spline) E4u closure; solves a 10×10 augmented RBF+poly system at construction (`gauss_solve`), parameterized by `sigma`; caches a 5×7 block. |
| `src/stencils/gaussian_E4u_1.cpp` | **Experimental** RBF (Gaussian-kernel) E4u closure parameterized by `epsilon`. |
| `src/stencils/multiquadric_E4u_1.cpp` | **Experimental** RBF (multiquadric-kernel) E4u closure parameterized by `epsilon`. |
| `src/operators/identity_stencil.hpp` | Test-only `detail::identity` stencil satisfying the `Stencil` concept; exercises operators without a real scheme. |
| `scripts/stencil_gen/output/E4_1.cpp` | **Not built.** Raw SymPy codegen output — the upstream of `src/stencils/E4_1.cpp`. Differs (no singularity guards); see "Relationship to the SymPy codegen". |

## Public API / entry points

### The data model
```cpp
struct info       { int p; int r; int t; int nextra; };  // boundary-block geometry
struct interp_info { int p; int t; };                      // wall-interpolation geometry
struct interp_line { std::span<const real> v; boundary left; boundary right; };
```
- **`p`** = interior half-width: the interior circulant stencil has `2p+1` points.
- **`r` × `t`** = the dense boundary block: `r` rows (closure equations) × `t` columns (support width).
- **`nextra`** = extra columns for Neumann data (only `E2_2`/`E4_2` use `nextra > 0`).
- `interp_info{p,t}` describes a wall interpolation block; **`{}` (all zeros) means the scheme does no cut-cell interpolation.**

### The `Stencil` concept (what each scheme struct must satisfy)
```cpp
template <typename T>
concept Stencil = requires(const T& st, bcs::type b, real h, real psi,
                           bool ray_outside, std::span<real> c, std::span<real> extra) {
    { st.query(b)        } -> std::same_as<info>;          // geometry for a given BC
    { st.query_max()     } -> std::same_as<info>;          // max geometry over all BCs (sizing)
    { st.query_interp()  } -> std::same_as<interp_info>;   // wall-interp geometry, or {} if none
    { st.nbs(h, b, psi, ray_outside, c, extra) } -> std::same_as<std::span<const real>>;
    { st.interior(h, c)  } -> std::same_as<std::span<const real>>;
};
```
A scheme struct typically also provides `interp_interior(y, c)` and `interp_wall(i, y, psi, c, right)` (the `stencil` wrapper requires them via its `any_stencil` vtable even though the concept does not; return the input span unchanged if unsupported). Each scheme additionally defines `constexpr int P, R, T, X` and splits `nbs` into `nbs_floating` / `nbs_dirichlet` / `nbs_neumann`.

### The type-erased handle (`ccs::stencils::stencil`, aliased to `ccs::stencil`)
Hand-rolled type erasure owning a raw `any_stencil*` (manual clone/delete, no smart pointer). Default-constructed is null — test with `explicit operator bool()`. Copy clones.
```cpp
info  query(bcs::type b) const;
info  query_max() const;
interp_info query_interp() const;

std::span<const real> nbs(real h, bcs::type b, real psi, bool ray_outside,
                          std::span<real> c, std::span<real> ex) const;
std::span<const real> interior(real h, std::span<real> c) const;
std::span<const real> interp_interior(real y, std::span<real> c) const;
std::span<const real> interp_wall(int i, real y, real psi, std::span<real> c, bool right) const;

// High-level cut-cell dispatch used by operators/derivative.cpp:
interp_line interp(int dir, int3 closest, real y,
                   const boundary& left, const boundary& right, std::span<real> c) const;

static std::optional<stencil> from_lua(const sol::table&, const logs& = {});
```

### Factory functions (declared in `stencil.hpp`)
```cpp
stencil make_E2_2();                              // 2nd deriv, also bound to second::E2
stencil make_E4_2();                              // 2nd deriv, also bound to second::E4
stencil make_E2_1(std::span<const real>);         // alpha[4]
stencil make_E4_1(std::span<const real>);         // alpha[2], cut-cell; throws if alpha[1] < 197/288
stencil make_E4u_1(std::span<const real>);        // alpha[2], uniform
stencil make_E6u_1(std::span<const real>);        // alpha[5], uniform
stencil make_E8u_1(std::span<const real>);        // alpha[7], uniform
stencil make_tension_E4u_1(real sigma);           // experimental RBF
stencil make_gaussian_E4u_1(real epsilon);        // experimental RBF
stencil make_multiquadric_E4u_1(real epsilon);    // experimental RBF
stencil make_polyE2_1(std::span<const real> floating_alpha,
                      std::span<const real> dirichlet_alpha,
                      std::span<const real> interpolant_alpha);

namespace second { extern stencil E2; extern stencil E4; }  // pre-built 2nd-deriv globals
void copy_zero_padded(std::span<const real> src, std::span<real> dst);  // helper
```

### `from_lua` dispatch table (authoritative — read this, not the old framework_architecture.md)
| `order` | `type` | Resolves to | Free params |
| --- | --- | --- | --- |
| `2` | `"E2"` | `second::E2` (E2_2) | — (alpha ignored) |
| `2` | `"E4"` | `second::E4` (E4_2) | — (alpha ignored) |
| `1` | `"E2"` | `make_E2_1(alpha)` | `alpha[4]` |
| `1` | `"E4"` | `make_E4_1(alpha)` (cut-cell) | `alpha[2]`, `alpha[1] >= 197/288` |
| `1` | `"E4u"` | `make_E4u_1(alpha)` | `alpha[2]` |
| `1` | `"E6u"` | `make_E6u_1(alpha)` | `alpha[5]` |
| `1` | `"E8u"` | `make_E8u_1(alpha)` | `alpha[7]` |
| `1` | `"tension_E4u"` | `make_tension_E4u_1(sigma)` | `sigma` (default `3.0`) |
| `1` | `"gaussian_E4u"` | `make_gaussian_E4u_1(epsilon)` | `epsilon` (default `0.9`) |
| `1` | `"multiquadric_E4u"` | `make_multiquadric_E4u_1(epsilon)` | `epsilon` (default `1.0`) |
| `1` | `"E2-poly"` | `make_polyE2_1(floating, dirichlet, interpolant)` | three alpha arrays (or one split array — see Gotchas) |

`order` defaults to `1` if omitted. There is **no** `order=4/6/8` branch; the order is encoded in the scheme name suffix (`_1` = first derivative, `_2` = second derivative), not the `order` field. Anything else logs an error and returns `std::nullopt`.

## How it works

**Coefficient generation contract.** Every scheme generates coefficients at *unit grid spacing* (`h=1`) and then divides through by `h` (first derivative) or `h*h` (second derivative) inside `interior`/`nbs`. For the `right` boundary of an odd-derivative scheme the block is additionally negated and reversed (`for v: v *= -1; std::ranges::reverse(c)`); even-derivative schemes only reverse. Callers pass a scratch `std::span<real>`; the method writes into a prefix and **returns a subspan view of it** (e.g. `c.subspan(0, R*T)`). Sizing is the caller's job — that is what `query_max()` is for.

**How operators consume a stencil** (`src/operators/derivative.cpp`):
1. `auto [p, rmax, tmax, ex_max] = st.query_max();` to size scratch buffers.
2. For each line: `st.interior(h, c)` fills the regular interior; `st.nbs(h, bc, psi, ray_outside, c, extra)` fills the boundary closure for the actual BC (`bc` from `bcs::type`).
3. For embedded objects, `st.interp(dir, closest, y, left, right, interp_c)` decides whether to use an *interior* interpolant (`interp_interior`) or a *wall* interpolant (`interp_wall`) based on how close the boundary/object is, and returns an `interp_line` carrying the coefficients plus the resolved left/right `boundary` stencil footprint. These coefficients are then assembled into the operator's `block`/`inner_block`/CSR matrices.

**Cut-cell vs uniform split.** Only `E2_1`, `E2_2`, `E4_2`, and `polyE2_1` return a non-empty `query_interp()` and implement real `interp_interior`/`interp_wall`. `E4_1` is psi-dependent in its *boundary closure* but returns `{}` for interp; all uniform (`E4u`/`E6u`/`E8u`) and all RBF schemes return `{}` and have no-op interp. Those latter schemes are for **uniform or boundary-fitted meshes** — point them at embedded geometry that needs interpolation and they will silently not interpolate.

**RBF schemes** (`tension`/`gaussian`/`multiquadric`) differ structurally: rather than hand-written closed-form coefficients, they build a small augmented RBF+polynomial linear system (collocation points `x = 0..5`, polynomial degree `q=3`, so a 10×10 system with 4 RHS) and solve it once at construction via in-file Gaussian elimination with partial pivoting (`gauss_solve<N,NRHS>`), caching a 5×7 boundary block. `nbs_floating` copies the cache; `nbs_dirichlet` drops the first (wall) row. Row 4 of the block is the classical E4 centered stencil. The `sigma`/`epsilon` shape parameter enters through the kernel functions (e.g. `tension_phi(r, sigma) = sigma|r| - 1 + exp(-sigma|r|)`).

## How to extend

To add a new scheme `foo_1`:
1. **Create `src/stencils/foo_1.cpp`** defining a `struct foo_1` that satisfies the `Stencil` concept. Copy an existing scheme of the same flavor as a template: `E4u_1.cpp` for a uniform closed-form closure, `tension_E4u_1.cpp` for an RBF closure, `E2_1.cpp`/`polyE2_1.cpp` for a cut-cell scheme with interpolation. Provide `constexpr int P/R/T/X`; `query`/`query_max`/`query_interp`; `interior(h,c)`; `nbs(...)` (typically split into `nbs_floating`/`nbs_dirichlet`/`nbs_neumann`); and `interp_interior`/`interp_wall` (return the input span unchanged if the scheme has no cut-cell interpolation). End with `stencil make_foo_1(...) { return foo_1{...}; }`.
2. **Declare the factory** `stencil make_foo_1(...);` in `src/stencils/stencil.hpp` near the other `make_*` declarations.
3. **Register the build target**: add `foo_1.cpp` to the `add_library(shoccs-stencils ...)` list in `src/stencils/CMakeLists.txt` and add `add_unit_test(foo_1 "stencils" shoccs-stencils)`.
4. **Wire `from_lua`**: add a branch in `src/stencils/stencil.cpp` keyed on `order` + `type` that reads any free parameters and calls your factory.
5. **Write `src/stencils/foo_1.t.cpp`** asserting `query`/`interior`/`nbs` coefficient values (compare against a known-good reference).

**For *derived* (not hand-written) schemes**, coefficients come from the SymPy pipeline in `scripts/stencil_gen/`, which emits to `scripts/stencil_gen/output/<name>.cpp`. The workflow is: regenerate → copy into `src/stencils/` → **manually re-add any hand-written guards** (see the E4_1 note below). See [stencil-pipeline](../../scripts/stencil_gen/docs/pipeline_reference.md) and the `stencil-pipeline` skill.

## Gotchas & invariants
- **`from_lua` only branches on `order==2` and `order==1`.** `order=2` returns the pre-built globals `second::E2`/`second::E4` (the 2nd-derivative E2_2/E4_2 used by the Laplacian/heat) and **ignores the alpha array**. All first-derivative schemes live under `order==1`.
- **`src/stencils/E4_1.cpp` is NOT a verbatim copy of `scripts/stencil_gen/output/E4_1.cpp`.** The production file adds hand-written singularity guards: it throws `std::invalid_argument` if `alpha[1] < 197/288` (≈0.684, where the interior polynomial denominator `D(psi)` hits zero) and clamps `psi` to `[1e-4, 1-1e-4]` (coefficients have poles at `psi=0` and `psi=1`). **Re-running codegen silently drops these guards — they must be re-added** (see commit `956d97c`).
- **Coefficient scaling is the scheme's responsibility, sizing is the caller's.** Coefficients are generated at `h=1`, then divided by `h` (or `h*h`); the `right` boundary block is negated+reversed (odd derivatives) or just reversed (even). The method writes into a caller-provided scratch span and returns a subspan; size scratch with `query_max()`.
- **Only `E2_1`/`E2_2`/`E4_2`/`polyE2_1` support cut-cell wall interpolation.** `E4_1`, all `E*u` uniform schemes, and all RBF schemes return `query_interp() == {}` and have no-op `interp_interior`/`interp_wall`. Using them with embedded geometry that needs interpolation will not interpolate.
- **`E2-poly` has a "dubious" optimizer convention** (`stencil.cpp:81-87`): if a *single* `alpha` array is supplied it is split as `subspan(0,6)` = floating, `subspan(6,9)` = dirichlet, `subspan(9,13)` = interpolant; otherwise the named `floating_alpha`/`dirichlet_alpha`/`interpolant_alpha` arrays are used. **Mismatched sizes read out of bounds.**
- **The `stencil` class is hand-rolled type erasure** owning a raw `any_stencil*` (manual clone/delete). The templated converting constructor (`stencil.hpp:156-161`) has unusual formatting — the requires-clause runs into the `stencil(` token. Do not "fix" it blindly.
- **RBF schemes are validated at a single shape parameter only.** They solve their linear system at construction in C++ but are only checked against one Python fixture value (sigma=3, eps=0.9, eps=1.0). Treat non-default `sigma`/`epsilon` as unvalidated.

## Maturity & known gaps
**Verdict: partial.** The core is mature — 11 scheme `.cpp` files compile into `shoccs-stencils`, all have registered Catch2 tests, and the `stencil` handle is a live production dependency: `from_lua` is called by all three concrete systems (`heat.cpp:88`, `scalar_wave.cpp:178`, `hyperbolic_eigenvalues.cpp:54`) and consumed by `operators/derivative.cpp`. The "partial" label reflects that **only three schemes appear in any checked-in production lua config**: `E2` (`heat.lua`, `order=2`), `E2-poly` (`scalar_wave.lua`/`eigenvalues.lua`, `order=1`), and `E4u` (`lua-configs/brady_livescu_4_3*.lua`, `order=1`). Everything else is wired into `from_lua` and unit-tested but config-orphaned.

Per-item status (from the verified audit flags):
- **`tension_E4u_1` / `gaussian_E4u_1` / `multiquadric_E4u_1`** — *experimental.* Complete, compiling, test-passing RBF research closures from git plan 42.5/42.6 (Apr 2026). Reachable via `from_lua` and exercised by the Python `brady2d_stability.py` optimization harness, but **no production config selects them** and each is verified at exactly one shape parameter. `multiquadric_E4u_1` additionally **failed the BL42 reflecting-layer stability test** (plan 44). Document as experimental; do not delete (they back live tests + the optimizer harness), do not treat as production. See [Cleanup Plan](../CLEANUP_PLAN.md).
- **`E6u_1`** — *mature but config-orphaned.* The audit refutes the "partial" label: fully implemented (complete interior + 5×8 floating / 4×8 dirichlet coefficient tables), reachable from the production `from_lua` path, passing end-to-end unit test, and cross-validated by the SymPy boundary pipeline (plan 20.3f). The only true gap is that no shipped `.lua` selects it. Keep as-is.
- **`E8u_1`** — *partial.* Real, fully implemented (`P=4,R=7,T=11`), unit-tested, and backed by a SymPy derivation pipeline (`p=4`). Config-orphaned with no system-level coverage — 8th-order boundary closures are stability-sensitive and unproven inside a time-stepping loop. The `known_limitations.md` "no Python derivation pipeline" note is stale (only the *sweeps* layer lacks E6/E8 params). Document as experimental/awaiting-demand; do not delete.
- **`E4_1` (cut-cell, `order=1 type=E4`)** — *mature.* The audit refutes "partial": fully built, dispatched, heavily unit-tested (golden values at psi=0.3/0.7/0.9 plus 9 singularity-guard sections), recently maintained, and the **sole provider of variable-psi cut-cell first-derivative coefficients**. The `known_limitations.md` "use E4u instead of E4" note is a narrowly-scoped optimizer-validation workaround (the BL feasible region `alpha1≈0.16` violates the legitimate `197/288` physics constraint), not a statement that E4_1 is broken. Gap: no end-to-end cut-cell simulation regression exercises it.
- **`from_lua` `order=2 type=E4` → `second::E4` (E4_2)** — *partial / available but not exercised end-to-end.* The `E4_2` stencil object is fully implemented and directly unit-tested, but the `from_lua` selection branch and config path have **zero coverage** (every `order=2` caller uses `type="E2"`). The sibling E2 path proves the machinery works. Cheap to lock down with one `laplacian.t.cpp` case.
- **`scripts/stencil_gen/output/E4_1.cpp`** — *experimental / regenerable codegen artifact (not built).* Intentionally tracked-but-not-promoted raw generator output; differs from the in-tree file by the deliberately-omitted singularity guards (the documented "copy then re-add guards" 27.6a workflow). Not the source of truth, not dead. Hazard: a new dev mistaking it for source — mitigated by a README/header note.

## Tests
- **11 Catch2 unit tests, label `"stencils"`**, one per scheme struct: `t-E2_1`, `t-E2_2`, `t-E4_1`, `t-E4_2`, `t-E4u_1`, `t-E6u_1`, `t-E8u_1`, `t-polyE2_1`, `t-tension_E4u_1`, `t-gaussian_E4u_1`, `t-multiquadric_E4u_1`. Run with `ctest --test-dir build -L stencils`.
- **Central/cut-cell (E2_1/E2_2/E4_1/E4_2) and polyE2_1** verify `nbs`/`interior`/`interp` coefficient values; `E4_1.t.cpp` (361 lines) adds golden checks at psi=0.3/0.7/0.9 and 9 singularity-guard sections (`REQUIRE_THROWS`/`NOTHROW` around the `alpha[1] >= 197/288` constructor guard, magnitude bounds, `D(psi)>0` positivity).
- **Uniform (E4u_1/E6u_1/E8u_1)** test the floating + dirichlet boundary closures at sample alpha, driven through `from_lua`.
- **RBF (tension @ sigma=3, gaussian @ eps=0.9, multiquadric @ eps=1.0)** assert the cached 5×7 block matches a Python fixture (`scripts/stencil_gen/tests/fixtures/{tension,gaussian,multiquadric}_e4u1_reference.py`) at **one parameter value only**, plus h-scaling / Dirichlet-row-drop / right-flip transforms. No parameter sweep, no cut-cell/interp coverage (those return `{}`).
- **Not covered in C++:** end-to-end use of E6u/E8u/E4-cut-cell/RBF inside a running system; the `order=2 type=E4` dispatch branch; the `stencil::interp()` dispatch logic (exercised only indirectly via `operators/derivative.t.cpp` and `laplacian.t.cpp`, which construct `stencils::second::E2` directly). No disabled/commented stencil tests.
- **Richer derivation correctness** lives in the Python suite (`scripts/stencil_gen`, ~1174 tests) — see the `stencil-testing` skill.
- **Note:** the C++ build is green (Kokkos 5.1 `create_graph` API break fixed 2026-06-04); `ctest` is 47/48. This subsystem owned `t-E2_1`, which is now FIXED: it was fixed by adding `.margin(1e-12)` to its `Approx` comparisons (the exact-rational E2 coefficients matched to ~15 digits, but a ~4e-16 roundoff value could not match an exact 0 without a margin). All 11 stencil tests pass. The only remaining failure is `t-laplacian` (see [Cleanup Plan](../CLEANUP_PLAN.md) §0a); `t-csr` was also fixed.

## Related docs
- [core-types.md](core-types.md) — `real`, `int3`, the `ccs` namespace conventions used throughout.
- [fields.md](fields.md) — `selection_desc` and the buffers that operator output lands in.
- **Operators** (derivative/gradient/laplacian) consume this subsystem — see their reference doc when available; `operators/boundaries.hpp` defines `bcs::type` and the `boundary` struct that `nbs`/`interp` take.
- [SymPy pipeline reference](../../scripts/stencil_gen/docs/pipeline_reference.md) and the `stencil-pipeline` / `stencil-sweeps` / `stencil-testing` skills — how derived coefficients are generated, swept, and tested offline.
- `plans/stencil-derivation-math-reference.md` — the math behind the cut-cell (TEMO) and polynomial constructions.
- `docs/handoff/known_limitations.md` — scheme-selection guidance (note: the "use E4u instead of E4" and "no Python pipeline for E6u/E8u" entries are partially stale per this audit).
- `docs/handoff/framework_architecture.md` (lines 214-219) — **legacy/stale**: its `from_lua` dispatch table is wrong (lists nonexistent `order=4/6/8` branches). Use the table in this doc instead. The `CLAUDE.md` "Stencils ... E2/E4/E6/E8 ... CSR cut-cell coefficients" line is also misleading — there is no CSR code in `src/stencils/`; cut-cell coupling is expressed via the `psi` argument + `interp_wall`, and CSR assembly lives in matrices/operators.
