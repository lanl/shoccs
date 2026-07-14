# Plan 48 — Cut-Cell Normal-Mode Stability: Generalized Kreiss, Reflection Coefficients, Two-Boundary Round-Trip, RK4 Containment

**Goal.** Give the Python pipeline a theorem-backed stability analysis of ψ-parameterized cut-cell closures — the Trefethen program (1983 determinant, 1984 reflection coefficients, 1985 two-boundary round-trip) applied to the closures the C++ solver actually time-steps — exposed as smooth, worst-cased-over-ψ margins ready for the optimizer.

**Execution model.** This plan is driven by the Workflow tool (see `plans/48-driver.workflow.js`), not ralph_wiggum. Items are sized for one implementer-agent pass each (one file, one concept), followed by an adversarial review agent that re-derives the math and runs the focused test command. All items run strictly sequentially (each commits before the next starts).

**Review status.** This plan passed a two-agent adversarial review (math + repo-fit); the corrections are folded in below. Items 2.3 and 3.3 had their original formulations *corrected* during review — implementers must follow the text below, and the 3.3 reviewer must still verify operator ordering against Trefethen 1985 Lemma 1 directly.

**Provenance.** Derived from the 2026-07-13 deep dive (12 adversarially verified gaps; see `docs/handoff` history and memory note `trefethen-gv-gap-analysis`). Key verified defects fixed here: L2 feeds only p of r closure rows; defective-κ masking (`min_singular_value` → inf); no ψ anywhere in the rigorous path; no reflection coefficients; no two-boundary analysis; no fully-discrete bridge.

---

## Constraints and non-goals

- **No time integration in Python.** All fully-discrete reasoning is resolvent-based (σ_min evaluations) or algebraic (root loci). Matrix powers `S^m`, `expm`, or any stepping of an update operator are out of bounds. RK4 runs only in C++ (Phase 4 L8 items).
- **The Dirichlet (wall-eliminated) block is the canonical analysis object** — it is what `derivative.cpp` time-steps (wall column stripped into CSR B). The floating block is never analyzed for stability.
- **Non-goals (later plans):** optimizer wiring of the new margins (bounds/constraints/fidelity ladder — plan 49), 2D transverse-wavenumber extension, staircase/Floquet, corners, the parabolic ν=2 program, polyE2_1 interp-row composition.
- **Scheme scope:** E2_1 end-to-end (fast derivation, ~0.03 s); E4_1 supported but every symbolic-derivation test marked `@pytest.mark.slow` (derivation > 170 s measured).

## Read first (all phases)

- `scripts/stencil_gen/stencil_gen/gks_kreiss.py` — whole module (565 lines). Note: `BoundaryRow = tuple[weights, col_offsets]` (:31), `KreissResult` (:49-82), `kappa_roots` (:90), `kreiss_matrix` (:178, requires `len(admissible) == len(boundary_rows)`), `min_singular_value` (:219, returns inf on `DefectiveKappaError`/`ValueError`), `make_s_grid` (:237), `kreiss_stability_check` (:422, `min_s_magnitude=0.1` mask).
- `scripts/stencil_gen/stencil_gen/brady2d_stability.py:359-435` — `_extract_stencil_data` (feeds first p rows only) and `layer2_kreiss_gks`.
- `scripts/stencil_gen/stencil_gen/temo.py:98-131` (`SchemeParams`), `:2424-2458` (`CutCellResult`: `floating` R×T, `dirichlet` (R−1)×T = rows 1..R−1, `alpha_symbols`, `dims=(r,t,R,T,X)`), `:911-928` (`build_cut_cell_deltas`: row-i deltas `[-(psi+i), -i, 1-i, …]` — only the wall delta is fractional).
- `scripts/stencil_gen/stencil_gen/group_velocity.py:718-813` (`psi_sweep_group_velocity` — the derive-per-call antipattern to avoid).
- `scripts/stencil_gen/stencil_gen/non_normality.py:347-416` (`pseudospectral_abscissa_estimate`, `kreiss_constant_estimate(L, s_grid)`).
- `scripts/stencil_gen/stencil_gen/cpp_bridge.py` (`BridgeResult` :29, `run_cpp_brady2d` :111, marker replacement :84-86, CSV parse :90-108), `lua-configs/brady_livescu_4_3.lua`.
- `src/stencils/stencil.cpp:36-80` (`from_lua` accepts `"E2"`, `"E4"` cut-capable types with alpha tables; `"E4u"`, `"tension_E4u"`, `"gaussian_E4u"`, `"multiquadric_E4u"`; `"E2-poly"`), `src/stencils/E4_1.cpp:18-40` (ψ poles; throws for `alpha[1] < 197/288`), `src/operators/derivative.cpp:305-442` (inner_block layout, wall column → CSR B).
- Trefethen 1983 (JCP 49) §for the determinant; Trefethen 1985 (Math. Comp. 45) Lemma 1 + Props 3-11 for round-trip; Reddy & Trefethen 1992 for pseudospectra-vs-RK-region; Kreiss & Wu 1993 for the resolvent bridge.

## Test conventions

All test commands run from `scripts/stencil_gen/` with `SYMPY_CACHE_SIZE=50000 uv run pytest … -x -q`. Mark anything deriving E4_1 symbolically or doing >10 eigendecomps `@pytest.mark.slow`. Cache derivations with `@pytest.fixture(scope="module")`. Use `pytest.approx` for float comparisons.

---

## Phase 0 — Substrate hardening (uniform path; independent value; prerequisite for everything)

### 0.1 Make defective/anomalous κ observable instead of "maximally stable"

**File:** `stencil_gen/gks_kreiss.py`, `tests/test_gks_kreiss.py`.

**Problem (verified empirically):** `min_singular_value` returns `np.inf` on `DefectiveKappaError`/`ValueError` (:229-232), so defective or root-count-anomalous s-points can never be selected as the witness; the orchestrator's defective-abort branches (:469-476, :504-514) are unreachable and an engineered double-root closure returns `is_stable=True, defective_kappa_detected=False`.

**Change:** add `min_singular_value_ex(...) -> tuple[float, str]` returning `(sigma, status)` with `status ∈ {"ok", "defective", "count_mismatch"}`; keep `min_singular_value` as a thin wrapper for compatibility. `_sweep_grid` and `_refine_witness` record anomalous points in a side list; `kreiss_stability_check` (a) sets `defective_kappa_detected=True` and runs `_classify_imag_axis` at any anomalous grid point on/near the imaginary axis, (b) never reports `is_stable=True` while anomalies are unclassified — verdict becomes `"defective"` (conservative fail) in that case. Do **not** map anomalies to σ=0 (would create false witnesses that Nelder-Mead chases); flag, classify, and fail-conservative.

**Test:** port the empirical reproduction — engineered stencil with a double admissible root κ=0.3 at s=0.4 (construction pattern already in `tests/test_gks_kreiss.py:254`): assert `defective_kappa_detected is True` and `is_stable is False`. Regression: all existing `test_gks_kreiss.py` verdicts unchanged.

**Run:** `pytest tests/test_gks_kreiss.py -x -q`

### 0.2 Generalized (augmented) Kreiss matrix for m closure rows ≥ p admissible roots

**File:** `stencil_gen/gks_kreiss.py`, `tests/test_gks_kreiss.py`.

**Problem (verified):** every real closure has more rows that differ from the interior than admissible roots (uniform RBF: 3 rows vs p=1 for E2, 4 vs p=2 for E4; cut-cell Dirichlet blocks: R−1 = 3 vs p=1 for E2_1, 4 vs p=2 for E4_1). The square `kreiss_matrix` (:178) raises `ValueError` → inf → those rows are silently dropped or the check is vacuous.

**Math (specify exactly; worked example below).** Sign convention: this module analyzes `u_t = −Σ_k a_k u_{n+k}` (the `kappa_roots` docstring convention) — state this in every new docstring; under the "raw operator weights" reading all w-signs flip. Let **p = number of admissible roots for Re(s) > 0 = L_left = −min(interior offsets)** (this equals the half-width only for centered interiors; all SHOCCS interiors are centered, but the worked example below is one-sided with L_left = 1, so use the L_left definition throughout). Left half-line, closure rows i = 0..m−1 at grid nodes n_i (default n_i = i), interior rows for n ≥ m. The interior recurrence constrains φ only on n ≥ m−p, so the general decaying solution is

  φ_n = Σ_{ℓ=1..p} α_ℓ κ_ℓ^n  for n ≥ m−p,  with φ_0..φ_{m−p−1} free nodal unknowns.

Unknown vector x = (φ_0, …, φ_{m−p−1}, α_1, …, α_p) ∈ C^m. Row i of M(s):

- nodal column q (q < m−p): `M[i,q] = s·δ_{n_i,q} + w_{i,q}` (note: the temporal δ is against the row's grid node n_i, not the row index i — they differ once `row_indices` is non-default; w_{i,q} = 0 if q not in row support);
- root column ℓ: `M[i, m−p+ℓ] = s·κ_ℓ^{n_i}·1[n_i ≥ m−p] + Σ_{j ≥ m−p, j ∈ support(i)} w_{i,j}·κ_ℓ^j`.

GKS eigensolution ⟺ det M(s) = 0 for some Re(s) ≥ 0 (imaginary-axis subtleties handled by the existing perturbation classifier). Reduces to the existing `kreiss_matrix` when m = p (regression-test this identity).

**Worked scalar example (include as a unit test).** Upwind interior `du_n/dt = −(u_n − u_{n−1})` (offsets [−1,0], weights [−1,1], h=1): κ(s) = 1/(1+s), p = 1. Two closure rows (m=2) ⇒ one nodal unknown φ_0, one amplitude α; M(s) is 2×2:
`M[0,0] = s + w_{00}`, `M[0,1] = Σ_{j≥1} w_{0j} κ^j`, `M[1,0] = w_{10}`, `M[1,1] = s·κ + Σ_{j≥1} w_{1j} κ^j`.
Choose w so that det M(s) has a known zero at s = 1 (mirror the existing engineered-unstable pattern at `test_gks_kreiss.py:427-454`) and assert the check finds it; choose a stable closure (e.g. row 1 = interior row) and assert stability.

**API:** `kreiss_matrix_general(interior_weights, interior_offsets, boundary_rows, s, *, row_indices=None) -> np.ndarray` (row_indices defaults to 0..m−1). `min_singular_value_ex`, `_sweep_grid`, `_refine_witness`, `kreiss_stability_check` gain a `row_indices` pass-through and dispatch to the general matrix whenever m ≠ p.

**Run:** `pytest tests/test_gks_kreiss.py -x -q`

### 0.3 Feed ALL differing closure rows to L2 (fixes the wrong-operator bug)

**File:** `stencil_gen/brady2d_stability.py` (`_extract_stencil_data` :359-390, `layer2_kreiss_gks` :393-435), `tests/test_gks_kreiss.py` (integration tests :689-800), `tests/test_brady2d_stability.py`.

**Change:** `_extract_stencil_data` extracts every row of D that differs from the (shifted) interior row — compare against the middle row within a tolerance — instead of the first p rows; return `(interior_weights, interior_offsets, boundary_rows, row_indices)`. `layer2_kreiss_gks` passes them to the generalized check. **Also update the duplicate test-local helper** `_extract_stencil_data_from_D` in `tests/test_gks_kreiss.py` (used at :709, :733) — it reproduces the first-p-rows bug independently, and the 0.3 recalibration only half-lands if it stays.

**Calibration (expected fallout — handle deliberately, do not force-fit):** verdicts for tension σ=3.0 (currently GKS-stable), Gaussian ε=0.1 (currently GKS-stable-but-eigenvalue-unstable), and the heuristic-vs-rigorous complementarity test may change now that the analyzed operator matches the real closure. For each flipped verdict: cross-check against L3 dense eigenvalues at N ∈ {20,40,80}; update the test with a comment citing this plan item; record the flip in the **Findings log** at the bottom of this plan.

**Run:** `pytest tests/test_gks_kreiss.py tests/test_brady2d_stability.py -x -q`

### 0.4 Replace the |s| < 0.1 disk mask with explicit near-origin treatment

**File:** `stencil_gen/gks_kreiss.py`, `tests/test_gks_kreiss.py`.

**Problem:** conservative (telescoping) closures have a structural determinant zero at s = 0 (κ = 1 consistency mode); the current fix masks the whole disk |s| < 0.1, excising exactly the z ≈ 1 borderline where generalized eigensolutions (Trefethen 1984 Thm 1b) live.

**Change:** shrink `min_s_magnitude` default to 1e-3 and, for every candidate witness with |s| < 0.1, discriminate the structural consistency zero **structurally — NOT via the perturbation classifier**. (Review finding: for any consistent derivative closure the κ=1 consistency root perturbs *inward* (κ ≈ 1−s since Σ k·a_k = 1), so `_classify_imag_axis` returns `outgoing_mode_detected` — its violation label — for perfectly stable closures, and a genuine violation returns the same label: zero discrimination.) Structural test: build the constant-mode coordinate vector x_const (nodal entries 1; amplitude 1 on the root continuing to κ=1 as s→0; other amplitudes 0) and classify the candidate as the consistency zero iff the σ_min right singular vector aligns with x_const (angle < tol) AND the closure rows annihilate constants (full-support row sums ≈ 0). Optionally corroborate by the vanishing order: fit σ_min(s) ~ |s|^ν along a ray into the RHP — derivative-consistent closures give ν ≈ 2; report the deflated margin σ_min(s)/|s|^ν. A genuine near-origin violation has a near-null vector NOT aligned with x_const → real witness.

**Test:** (a) existing Godunov–Ryabenkii s=0 reduction test still passes; (b) construct a closure with an engineered determinant zero at s = 0.05 whose null vector is not the constant mode (same reverse-engineering pattern as 0.2's example) and assert it is caught; (c) a conservative/telescoping closure's near-origin candidate classifies as consistency-zero, not violation; (d) E2/E4 classical and tension-σ=3.0 verdicts unchanged (post-0.3 baselines).

**Run:** `pytest tests/test_gks_kreiss.py -x -q`

### 0.5 Witness-search robustness for the augmented Kreiss matrix (added 2026-07-13 after item 1.2 blocked)

**File:** `stencil_gen/gks_kreiss.py`, `tests/test_gks_kreiss.py` (tests may import `make_cutcell_evaluator` from 1.1; production changes stay in gks_kreiss.py). **Runs after 1.1; blocks 1.2.**

**Problem (diagnosed by the blocked 1.2 attempt; dense eigenvalues at N ∈ {20,40,80,160} confirm E2_1 α=0 is stable at all five test ψ, max Re λ ≤ 8e-6, so these are all false-positive mechanisms):**
1. For augmented m > p closures, σ_min(M(s)) → 0 as |s| → ∞ (measured σ_min·|s| → 1/4 for E2_1) while det M(s) → ∞ — root columns scale like κ^j ~ (2|s|)^{−j}, so the σ_min proxy structurally diverges from the determinant condition at large |s| and the grid argmin lands at the corner. (The square m = p matrix grows like |s| — no artifact, which is why 0.1–0.4 tests never saw this.)
2. `_refine_witness` Nelder-Mead is unbounded and descends that slope to |s| ~ 1e153–1e155 where σ underflows to exactly 0 → `is_stable=False`, verdict `no_candidates` (ψ ∈ {1e-3, 0.1}; generic in α).
3. NM probes points at Re(s) ~ 1e-13 on the E2 sub-sonic axis segment (|ω| < 1: both κ roots exactly unimodular → count_mismatch anomalies); `_triage_anomalies` sends these to `_classify_imag_axis`, which item 0.4 established has zero discrimination near the origin → deterministic false `outgoing_mode_detected` (ψ = 0.9).
4. Item 0.4's structural test covers only the conservative x_const = (1,…,1, e_{κ→1}) zero. The wall-eliminated Dirichlet cut block has a **different** structural double zero at s = 0 for all ψ: every closure row annihilates *tail* constants (Σ_{j ≥ m−p} w_ij ≡ 0 identically in ψ), null vector (0_nodal, e_{κ→1}) — rows do NOT annihilate full constants, so `_classify_near_origin` rejects it; at ψ = 0.9999 the ψ→1 row-coincidence degeneracy shrinks σ_min at the |s| = 1e-3 barrier below sigma_tol → false `near_origin_violation`.

**Change:**
(a) **Column-equilibrate** M(s) before the SVD (scale each column to unit ∞-norm; det-zero locations are invariant since det(MD) = det(M)·det(D) with det D > 0; σ_min becomes a meaningful relative margin at all |s|). **Bound the witness search** to the provable eigensolution region: any GKS eigensolution with a decaying tail attains its sup at some node k, giving |s| ≤ S_max := max over all rows (closure and interior) of Σ_j |w_ij| (derive this sup-norm/Gershgorin bound in the docstring); mask grid points and clamp/reject NM iterates beyond |s| ≤ S_max (+ small margin).
(b) Clamp NM probe points to Re(s) ≥ eps_imag (consistent with `make_s_grid`'s strip) — reflect, don't evaluate, below it.
(c) Route near-origin anomalies (|s| < 0.1) through the structural discrimination of 0.4/(d) below, never `_classify_imag_axis`.
(d) **Generalize the structural benign-zero test:** classify a near-origin σ_min dip as benign iff the right singular vector aligns (angle < tol) with x_const = (1,…,1, e_{κ→1}) [conservative full-constant zero, row sums ≈ 0] OR with x_tail = (0_nodal, e_{κ→1}) [Dirichlet tail-constant zero, Σ_{j ≥ m−p} w_ij ≈ 0], with the matching row-sum identity verified. Handle the double zero: fit the vanishing order ν ∈ {1, 2} along a ray into the RHP and report the deflated margin σ_min(s)/|s|^ν; the ψ→1 degeneracy must not trip sigma_tol.

**Tests:** (i) the 1.2 diagnostic as regression: E2_1 α=0 Dirichlet rows via `make_cutcell_evaluator` at ψ ∈ {1e-3, 0.1, 0.5, 0.9, 0.9999} → `kreiss_stability_check(..., row_indices=…)` returns `is_stable=True` at all five, cross-checked in-test against dense eigenvalues at N=40; (ii) large-|s|: equilibrated σ_min at |s| = 1e3 is O(1), not O(1/|s|); (iii) the 0.2 engineered det-zero at s=1 still caught post-equilibration; (iv) returned `witness_s` always satisfies |s| ≤ S_max; (v) all 0.1–0.4 tests green unchanged.

**Run:** `pytest tests/test_gks_kreiss.py tests/test_cutcell_eval.py -x -q`

---

## Phase 1 — Cut-cell ingestion (item 1)

### 1.1 Cached lambdified cut-cell evaluator

**File:** new `stencil_gen/cutcell_eval.py`, new `tests/test_cutcell_eval.py`.

**Why first:** every downstream item consumes closure rows at (ψ, α) points; deriving per call is fine for E2_1 (~0.03 s) but measured > 170 s for E4_1 — unusable in any sweep or optimizer loop.

**API:**
```python
@functools.lru_cache(maxsize=None)
def make_cutcell_evaluator(scheme: SchemeParams, bc: str = "dirichlet") -> CutCellEvaluator
```
Internally: dispatch by `scheme.zeros` exactly as production does (`derive_cut_cell_mathematica` for E4_1, `derive_cut_cell_scheme` otherwise); `sympy.lambdify` every entry of the chosen block over `(psi, *alpha_symbols)` once. `SchemeParams` must be hashable — if it isn't frozen, key the cache on `(p, q, s, nextra, nu, zeros)`.

`CutCellEvaluator` exposes:
- `.rows(psi: float, alpha: dict) -> tuple[list[BoundaryRow], np.ndarray]` — the (R−1) Dirichlet closure rows with the **wall column dropped** (homogeneous Dirichlet is the correct GKS setting; the determinant condition tests the homogeneous resolvent problem), integer column offsets renumbered to grid nodes 0..T−2, and `row_indices = [0..R−2]` (dirichlet row i sits at grid node i);
- `.dims`, `.alpha_symbols`;
- `.denominators` — the set of sympy denominator polynomials in (ψ, α) collected from the block entries, for feasibility pre-checks;
- `.interior() -> (weights, offsets)` — the matching interior stencil from `interior.py`.

**Tests:** (a) ψ=1 identity: `rows(1.0, alpha)` matches the uniform boundary rows (the repo already asserts cut-cell ψ=1 ≡ uniform to 1e-10 — reuse that construction from `tests/test_group_velocity.py:740-793`); (b) cache hit: second `make_cutcell_evaluator(E2_1)` returns the same object; (c) 20-ψ evaluation after warm cache completes in < 0.1 s (generous bound; measured 0.09 ms); (d) E4_1 variant `@pytest.mark.slow`.

**Run:** `pytest tests/test_cutcell_eval.py -x -q`

### 1.2 ψ-parameterized Kreiss check *(precondition: item 0.5 — a first attempt without it blocked with deterministic false-unstable verdicts at 4 of 5 ψ)*

**File:** `stencil_gen/cutcell_eval.py` (or new `stencil_gen/cutcell_normal_modes.py` if cutcell_eval.py exceeds ~300 lines), tests in `tests/test_cutcell_eval.py`.

**API:** `kreiss_cutcell_check(scheme, alpha: dict, psi: float, **kreiss_kwargs) -> KreissResult` — assemble rows via the evaluator, call `kreiss_stability_check` with `row_indices`. Validate m = R−1 rows, p admissible roots, m−p nodal unknowns (E2_1: 3 = 2+1; E4_1: 4 = 2+2).

**Feasibility guard:** before assembly, evaluate `.denominators` at (ψ, α); if any |denominator| < 1e-12, return a distinct `infeasible` marker (not a stability verdict). Denominator vanishing is a *parameterization* failure and must never be conflated with GKS instability.

**Tests:** E2_1, α = 0 (the plan-35 baseline configuration): stable for ψ ∈ {1e-3, 0.1, 0.5, 0.9, 0.9999}. Cross-check one ψ against dense eigenvalues of the assembled half-line operator at N=40 (builder arrives in 3.1; until then use the test-local assembly already present in `tests/test_group_velocity.py:1030-1050`).

**Run:** `pytest tests/test_cutcell_eval.py -x -q`

### 1.3 Worst-case-over-ψ margin

**File:** same module, tests same file.

**API:** `kreiss_margin_over_psi(scheme, alpha, psi_grid=None) -> CutCellPsiSweep` (do not name it `PsiSweepResult` — that name already exists in `group_velocity.py`) with fields `worst_sigma_min`, `worst_psi`, `any_unstable`, `infeasible_psi: list`, `per_psi: dict`. Default grid clustered near 0 (ILW literature: ψ→0 is the critical regime) and matched to the C++ clamp `[1e-4, 1−1e-4]`:
`{1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2} ∪ Chebyshev-11 on [0.05, 0.95] ∪ {0.99, 0.999, 0.9999}`.

**Run:** `pytest tests/test_cutcell_eval.py -x -q -k "psi_sweep or margin"`

### 1.4 Calibration / control experiment (findings item — no production code)

**Provenance note on 197/288 (established 2026-07-13):** E4_1's `alpha[1]` is NOT a Brady–Livescu boundary alpha — per `temo.py:3160-3176` the zeros-path free parameters are `alpha_2 → alpha_0` (boundary shape) and the free conservation quadrature weight `w_4 → alpha_1`. The 197/288 floor is a denominator-positivity condition on **w_4**, introduced by commit `15c0ddd` (plan 26.6-followup-d5b), from D(ψ) = 288·w₄ + 648·ψ + 90·ψ² + 12·ψ³ − 197 with D′ > 0, so D > 0 on (0,1) ⟺ D(0) = 288·w₄ − 197 ≥ 0. It is a parameterization-singularity constraint specific to the `zeros=(3,4)` + `w_0=ψ` conservation solve — not a stability constraint, and not comparable to the optimizer's uniform E4 alphas (see plan-43 category error under Out of scope). NOTE: the current `E4_1.cpp:22-24` comment records a *different* degree-6 polynomial with D(0) = 1728·α₁ − 1182 (same floor, 6× the constant) — the two records disagree about D itself.

Script under `scripts/stencil_gen/` run once by the workflow (not committed as a test): (a) E2_1 α=0 across the full ψ grid — expect stable everywhere (validates against plan-35 history); (b) E4_1 at `alpha[1] ∈ {0.60, 197/288 + 0.05}` with `alpha[0]` at the test-suite default 0.1 (these are E4_1's OWN (alpha_2, w_4) coordinates — do not reuse optimizer/BL alphas here) — record verdict, worst ψ, and whether the 0.60 case is flagged infeasible (denominator) vs unstable `[slow]`; (c) the same configurations through 40-point dense eigenvalues as cross-check; (d) **re-derive the denominator floor independently**: collect `CutCellEvaluator.denominators` for E4_1, isolate the ψ-interior-vanishing factor D(ψ; w₄), verify (or correct) the 197/288 floor symbolically, and reconcile the commit-15c0ddd cubic vs the E4_1.cpp degree-6 comment — record which is the true denominator. Append results to the **Findings log**. Acceptance: no contradiction between Kreiss verdict and dense eigenvalues at matched ψ; contradictions block Phase 2 until explained.

---

## Phase 2 — Reflection coefficients R(s; ψ) (item 2)

### 2.1 Mode partition

**File:** new `stencil_gen/reflection.py`, new `tests/test_reflection.py`.

**API:** `mode_split(interior_weights, interior_offsets, s) -> ModeSplit` with `rightgoing` (|κ| < 1 − tol), `leftgoing` (|κ| > 1 + tol), `borderline` (unit-modulus, classified by the existing perturbation test from `_classify_imag_axis` — factor that logic into a shared helper rather than duplicating). For centered 2p+1 interiors and Re(s) > 0: exactly p rightgoing, p leftgoing (assert; degree-2p characteristic polynomial). Evaluate on the imaginary axis at `s = iω + eps_imag` (consistent with `make_s_grid`'s strip) so borderline roots are resolved by the half-plane limit.

**Run:** `pytest tests/test_reflection.py -x -q -k mode_split`

### 2.2 Reflection matrix and margin

**File:** `stencil_gen/reflection.py`.

**Math (specify exactly).** With m closure rows, nodal unknowns φ_0..φ_{m−p−1}, rightgoing amplitudes α ∈ C^p, and a unit-amplitude incident leftgoing mode μ_k (|μ_k| > 1):

  φ_n = μ_k^n + Σ_ℓ α_ℓ κ_ℓ^n (n ≥ m−p), φ_q free (q < m−p).

The m closure equations read `M(s)·x = −b_k(s)` with M(s) from item 0.2 and

  `b_k[i] = s·μ_k^{n_i}·1[n_i ≥ m−p] + Σ_{j ≥ m−p, j ∈ support(i)} w_{i,j}·μ_k^j`.

Solve for x; the reflection matrix column is `R[:,k] = α`-block of x. R(s) is p×p. Margin:

  `max_reflection = max over the imaginary-axis s-grid of ‖R(s)‖₂`.

**Properties to unit-test (these are the reasons R beats σ_min):**
- **Normalization invariance:** R(s) is invariant under scaling an assembled *equation* — i.e., multiply row i of [M(s) | b_k(s)] jointly by 10 and R is exactly unchanged. Implement the test at that level. (Caution from review: scaling only the stored `BoundaryRow` *weights* does NOT preserve R, because the temporal coefficient stays 1 — the spatial parts of M and b scale but the s-parts don't. Do not write the test that way.) σ_min(M) changes under the same joint row scaling — assert both to document the contrast.
- **Singularity equivalence:** at an engineered det M(s₀)=0 closure (from 0.2's worked example), ‖R(s)‖ → ∞ as s → s₀ while the GKS check flags s₀.
- **Analytic anchor:** first-order upwind has characteristic degree 1 — one admissible root, **zero leftgoing modes**, so R is p×0 and there is nothing to reflect: keep upwind only as an assert-R-is-empty case. The hand-derivable anchor is the **E2 central interior** (κ roots analytic) with one one-sided closure row: R is 1×1 — derive R(s) by hand in the test comment and assert against it.
- **Sonic-point behavior:** near s where interior roots coalesce on the unit circle (E2: s = ±i), expect ‖R‖ ~ O(eps^{−1/2}) spikes at the eps_imag offset — these must route through the defective/borderline path (0.1), not report as clean values.

**API:** `reflection_matrix(interior_weights, interior_offsets, boundary_rows, s, *, row_indices=None) -> np.ndarray`; `max_reflection(..., s_grid=None) -> ReflectionResult` (fields: `max_norm`, `argmax_s`, `finite: bool`).

**Run:** `pytest tests/test_reflection.py -x -q`

### 2.3 Right-end closures

**File:** `stencil_gen/reflection.py`, tests same file.

**Convention (CORRECTED during review; sign errors are the #1 historical pain point).** The coordinate mirror n → N−n′ is **pure index reversal, no negation**: interior weights map `a_k → a_{−k}` (offsets negated, weights unchanged), which maps κ → 1/κ; closure rows likewise get offsets negated and node indices remapped to distance-from-the-right-end. Do **NOT** reverse-and-negate: revneg gives κ′(s) = 1/κ(−s) — for the upwind toy revneg yields κ′ = 1−s while the true mirror is 1/κ = 1+s, so the original formulation of this item was wrong. (The negation belongs to *codegen's construction* of physical right-wall coefficients from left-wall ones — a statement about the physical rows, not about the coordinate mirror.) Consequently: if you start from the **physical right-end rows**, apply pure reversal and reuse `reflection_matrix`; if you start from the **stored left-end rows** and rely on the codegen revneg symmetry to represent the physical right end, the composite map is: interior reversed, closure weights negated, offsets unchanged. Implement `mirror_closure(boundary_rows, interior, *, from_physical_right: bool)` covering both entry points, and document which one each caller uses. **Worked checks (unit tests):** (a) upwind toy — assert the mirrored interior's root is analytically 1/κ = 1+s; (b) antisymmetric interior + symmetric closure (E2 classical, α=0) — assert **norm equality** ‖A_R(iω)‖ = ‖A_L(iω)‖ on the axis (the entrywise identity A_R(iω) = conj(A_L(iω)) holds only after canonical ordering of the mode basis — test norms, not entries, unless you implement canonical root ordering).

**Run:** `pytest tests/test_reflection.py -x -q -k mirror`

### 2.4 ψ-swept reflection margin + report wiring

**File:** `stencil_gen/reflection.py` (sweep), `stencil_gen/brady2d_stability.py` (report field only), tests both.

`reflection_over_psi(scheme, alpha, psi_grid=None, side="left") -> CutCellPsiSweep` (reuses 1.3's grid and infeasibility guard). Report wiring: add an **opt-in** `layer_cutcell: dict | None` field to `StabilityReport` (default None; populated only by a new entry point `cutcell_stability_report(scheme, alpha, *, psi_grid=None)` — do NOT insert into the L1–L8 cascade in this plan; optimizer wiring is plan 49). Keys: `worst_kreiss_sigma`, `worst_psi`, `max_reflection`, `argmax_psi`, `reflection_product` (filled by 3.3), `infeasible_psi`.

**Do NOT touch `_FIELD_LAYER_ALIAS` or `make_objective`** (review finding: the alias table maps dotted-path heads to cascade layer ints consumed by `_infer_max_layer` → `brady2d_stability_score`; a non-cascade layer is unrepresentable there — None makes `make_objective` raise, any int runs a cascade that never populates `layer_cutcell`). `extract_field`'s dotted-path traversal already handles `layer_cutcell.max_reflection` on any report object — add exactly that unit test in `tests/test_optimizer.py` (fits the existing `TestExtractField` class). Objective/alias wiring is plan 49. Naming rule: every new test in `test_reflection.py` that exercises the sweep/report must contain `cutcell` in its name so `-k` filters select it.

**Run:** `pytest tests/test_reflection.py -x -q` and `pytest tests/test_optimizer.py::TestExtractField -x -q`

### 2.5 Calibration: the Gaussian ε=0.1 case gets a mechanism

The repo's documented GKS-passes-but-eigenvalue-unstable case (uniform Gaussian ε=0.1, `test_gks_kreiss.py:689-748`). Compute `A_L`, `A_R` for that closure: expected outcome — each end individually GKS-stable (finite ‖R‖) with `‖A_L‖·‖A_R‖ > 1` on part of the imaginary axis, i.e. the two-boundary mechanism, attributed analytically for the first time. Add as a (fast) test asserting `max_reflection` finite and the product > 1; append numbers to the Findings log. If the product is NOT > 1, that falsifies the round-trip explanation for this case — record it and re-examine at Phase 3.3 with the full E_Δj criterion before proceeding.

**Run:** `pytest tests/test_reflection.py -x -q -k gaussian`

---

## Phase 3 — Two-boundary / short-cut-line tier (item 3)

### 3.1 Interval-operator assembler

**File:** new `stencil_gen/interval_operator.py`, new `tests/test_interval_operator.py`.

**API:** `build_interval_operator(scheme, alpha, psi_L, psi_R, N) -> np.ndarray` — the N×N semi-discrete operator for a fluid interval with cut closures at both ends, mirroring `derivative.cpp`'s `[dense_left | circulant | dense_right]` layout with wall columns dropped and right rows produced by the 2.3 mirror convention. `psi_* = 1.0` means a uniform domain boundary at that end (matching the C++ convention of passing psi=1.0 to `nbs`).

**Tests:** (a) `psi_L = psi_R = 1` reproduces `_build_classical_diff_matrix` / the uniform builder exactly; (b) row-sum consistency (each row annihilates constants) at several (ψ_L, ψ_R); (c) N large: interior rows equal the circulant stencil; (d) smallest assemblable N (both closures overlapping) either assembles consistently or raises a clear `IntervalTooShort` — decide by construction, document the minimum N = m_L + m_R (closure rows may not overlap; below that the line has no interior row and the operator is all-closure — still assemble it, it is exactly the dangerous case, but require column supports to fit: raise if t_L + t_R − overlap > N with a precise message).

**Run:** `pytest tests/test_interval_operator.py -x -q`

### 3.2 Cheap tier: resolvent-based interval scan

**File:** `stencil_gen/interval_operator.py`.

**API:** `interval_stability_scan(scheme, alpha, *, psi_pairs=None, N_values=(m_min, m_min+2, m_min+5, m_min+10, 20, 40)) -> IntervalScanResult`: per (ψ_L, ψ_R, N): `max Re λ` (dense eig — matrices are ≤ 40×40, microseconds) and `kreiss_constant_estimate(L, s_grid)` on a refined near-imaginary-axis grid (NOT the coarse 8×12 default — pass an explicit grid). **Bright line restated: no matrix powers, no stepping; resolvent quantities only.** Default `psi_pairs`: the 1.3 grid × itself, thinned to ~50 pairs (corners + diagonal + random-seeded Latin sample with fixed seed).

**Run:** `pytest tests/test_interval_operator.py -x -q -k scan`

### 3.3 Rigorous tier: round-trip criterion (Trefethen 1985 Lemma 1)

**File:** `stencil_gen/reflection.py`, tests in `tests/test_reflection.py`.

**Math (ordering CORRECTED during review).** At s on the imaginary-axis grid with mode split (K = diag rightgoing κ, |κ| ≤ 1; Mdiag = diag leftgoing μ, |μ| ≥ 1) and per-end reflection matrices A_L(s; ψ_L), A_R(s; ψ_R), the round-trip cycle applies (right-to-left) A_L first, then rightgoing transit, then A_R, then leftgoing return:

  `E_Δj(s) = Mdiag^{−Δj} · A_R(s; ψ_R) · K^{Δj} · A_L(s; ψ_L)`  (any cyclic rotation is similar and has the same spectrum; the *original draft's* Mdiag^{−Δj}·A_L·K^{Δj}·A_R is NOT a cyclic rotation of this and is wrong. The implementer and reviewer must both verify roles/ordering against Trefethen 1985 Lemma 1 directly.)

Conventions to pin (an off-by-one rephases E): document which reference nodes K^{Δj} propagates between, define Δj accordingly, and require closure supports not to overlap (Δj ≥ support extent — tie to 3.1's `IntervalTooShort` minimum).

Criteria: GKS-eigensolution of the pair ⟺ some eigenvalue of E_Δj equals 1. **Do not assume monotonicity in Δj — it is false** (on the axis K and Mdiag are pure phases; ρ(E_Δj) oscillates with Δj, which is exactly the physics of Δj-specific resonances). Report `round_trip_radius = max over the s-grid AND over Δj ∈ {Δj_min, …, Δj_min+5} of ρ(E_Δj(s))`, plus the decoupled screen `reflection_product = max_s ‖A_L‖·‖A_R‖` (which upper-bounds ρ via ‖K‖ = ‖Mdiag^{−1}‖ = 1 on the axis).

**Ordering tripwire (required):** at p=1 every factor is a scalar, so ANY ordering gives the same ρ and the acceptance tests below cannot catch an ordering bug. Add a p=2 cross-validation: for a p=2 configuration (E4 classical uniform, or E4_1 at ψ_L ≠ ψ_R `[slow]`) where the 3.2 dense scan finds an unstable interval eigenvalue s*, assert ρ(E_Δj(s*)) ≈ 1 at the matching Δj — a wrong ordering fails this.

**Acceptance test:** the Gaussian ε=0.1 pair (ψ=1 uniform ends) must be flagged (`round_trip_radius > 1` or product > 1) at the Δj corresponding to the N=20 case where L3 catches it — the analytic tier now explains the repo's documented false negative. E2_1 α=0 at several (ψ_L, ψ_R): radius < 1.

**Run:** `pytest tests/test_reflection.py -x -q -k round_trip`

### 3.4 Wire into `layer_cutcell` + docs

Fill `reflection_product`, `round_trip_radius`, `interval_worst_abscissa` in the 2.4 report; write `scripts/stencil_gen/docs/cutcell_normal_mode_reference.md` (math, API, calibration results, limits: 1D, semi-discrete, straight cuts) and update `.claude/skills/group-velocity-analysis/SKILL.md`'s function table (add the new entries; mark `cut_cell_group_velocity`/`psi_sweep_group_velocity` as superseded-for-stability, retained as dispersion diagnostics).

**Run:** `pytest tests/ -x -q -k "cutcell or reflection or interval"` (fast subset)

---

## Phase 4 — Fully-discrete containment + C++ L8 upgrades (revised item 4)

**Division of labor (user constraint):** Python never time-steps. Python certifies (a) the semi-discrete GKS/reflection margins (Phases 0–3) and (b) resolvent containment against the RK4 stability region (Kreiss–Wu / Reddy–Trefethen route). C++ is the only place RK4 runs — and is upgraded so it can actually play that role for cut cells.

### 4.1 RK4 stability-region boundary containment (resolvent only)

**File:** new `stencil_gen/fully_discrete.py`, new `tests/test_fully_discrete.py`.

**Math.** RK4 amplification `R(ζ) = 1 + ζ + ζ²/2 + ζ³/6 + ζ⁴/24`; stability region S = {ζ : |R(ζ)| ≤ 1}. Compute the boundary ∂S once (march the level set |R(ζ)|=1 from ζ=0 by solving R(ζ) = e^{iθ} over θ ∈ [0, 2π) with Newton continuation; cache the polyline at module level — pure numpy, no stepping of any operator). For an operator L (uniform or interval) and dt = CFL·h:

  `containment_margin(L, dt) = min over z ∈ ∂S of σ_min(zI − dt·L)`  (and assert all eigenvalues of dt·L lie in S; report both).

Interpretation (document in module docstring): eigenvalues-in-S is necessary; the joint criterion is stronger than "within an ε-neighborhood": **eigs-inside + containment_margin ≥ ε certifies Λ_ε(dt·L) ⊂ S outright**, because every connected component of the ε-pseudospectrum contains an eigenvalue, so no component can lie wholly outside S, and margin ≥ ε forbids crossing ∂S. By Reddy–Trefethen this is what controls fully-discrete power-boundedness for non-normal L. Together with the semi-discrete GKS margin this instantiates the Kreiss–Wu hypothesis chain (resolvent-stable semi-discrete + locally stable one-step method); note explicitly in the docstring that Kreiss–Wu covers the half-line problem, not marginal fixed-Δj two-boundary cases — those remain covered by 3.2/3.3 plus the C++ rung. **Discretization honesty:** σ_min(zI − dt·L) is 1-Lipschitz in z, so report the certified value `margin_certified = margin_grid − max_arc_spacing/2`; default `n_boundary = 1024` (at 256 points the spacing correction ≈ 0.05 is material for small margins).

**API:** `rk4_containment(L: np.ndarray, dt: float, *, n_boundary=1024) -> ContainmentResult` (`margin`, `margin_certified`, `eigs_inside: bool`, `argmin_z`). Add `rk4_containment_over_psi(scheme, alpha, cfl, h, psi_grid)` reusing 3.1's assembler at a fixed representative N (default 40), **plus** an evaluation at the worst (ψ_L, ψ_R, N) triple reported by the 3.2 scan — a fixed N can miss N-specific resonances.

**Tests:** (a) L = −I (scalar decay): margin equals the analytic distance from ∂S to the point −dt (hand-derivable — actually σ_min(zI − dt·L) = |z + dt|; assert against `min |z + dt|` over the polyline); (b) a normal advection circulant at CFL 0.8: eigenvalues inside, margin > 0; at CFL 3.0: `eigs_inside is False`; (c) scale test — margin halves when a non-normal perturbation doubles the resolvent norm (qualitative: assert margin decreases).

**Run:** `pytest tests/test_fully_discrete.py -x -q`

### 4.2 `transient_growth_bound` honesty fix

**File:** `stencil_gen/non_normality.py` (:~535 and docstring :54-55), `stencil_gen/brady2d_stability.py` gate comment, affected tests.

`tgb = e·K` is documented as an upper bound on sup‖e^{tL}‖ but the Kreiss matrix theorem gives K ≤ sup ≤ e·n·K, and K itself is a coarse-grid *underestimate*. Rename field to `transient_growth_estimate` (keep `transient_growth_bound` as a deprecated alias property so dotted paths and persisted JSON keep working), fix docstrings to state "lower-bound-flavored estimate: e × grid-sampled Kreiss constant; rigorous upper bound would be e·n·K", and leave the gate threshold unchanged (it is calibrated, not derived).

**Run:** `pytest tests/test_non_normality.py tests/test_brady2d_stability.py -x -q`

### 4.3 L8 cut-cell capability — investigation gate

**File:** findings item, no production code. **Blocks 4.4.**

Read `src/systems/scalar_wave.cpp` (+ `from_lua`) and one simulation smoke test to answer: does `scalar_wave` support embedded shapes end-to-end (BC application at object boundaries, `logs/system.csv` output unchanged)? The 2D "cycle" smoke tests use spheres with heat/partial-Euler; scalar-wave-with-shapes is unverified. Outcome A (supported): proceed to 4.4 with `system = "scalar wave"`. Outcome B (not supported): 4.4 falls back to `type = "heat"` for the cut-cell template (parabolic — validates ψ-assembly/CSR paths and blow-up behavior but NOT the hyperbolic mechanism; record the limitation in the Findings log and file the scalar-wave-shapes work as a follow-up plan item). Append the determination + file:line evidence to the Findings log.

### 4.4 L8 upgrades: embedded-shape template, growth-rate metric, cut-capable dispatch

**Files:** new `lua-configs/brady_livescu_4_3_cut.lua`; `stencil_gen/cpp_bridge.py`; `stencil_gen/brady2d_stability.py` (`_L8_SCHEME_TYPE` + `layer8_cpp_simulation`); `tests/test_cpp_bridge.py`.

Sub-items (sequential, same files):
1. **Template:** copy `brady_livescu_4_3.lua`, add a `shapes = { { type = "sphere", center = …, radius = --{{RADIUS}}-- } }` block (pick center/radius so ψ values span near-0 to near-1 across N ∈ {31, 61} — compute the realized ψ set in the test and assert it has points < 0.05 and > 0.95), add `--{{CFL}}--` marker. Keep the original template untouched (uncut L8 stays the regression baseline).
2. **Growth rate:** `BridgeResult` already carries the full `t_trace`/`linf_trace` series (cpp_bridge.py:90-108) — compute `growth_rate: float | None` from them: slope of a least-squares fit of log(L∞) over the trailing half of the series (None if any non-finite or if L∞ < 1e-12 throughout); `final_linf` keeps its meaning. Stability verdict for the cut rung: `growth_rate < rate_tol` (default 0.05, calibrate in sub-item 5) rather than the blow-up-only `final_linf < 10`.
3. **Dispatch:** do NOT mutate `_L8_SCHEME_TYPE` (its `("E4","classical") → "E4u"` entry is the uniform-validation path and would collide on the same key). Add a separate `_L8_CUT_SCHEME_TYPE` table — `("E2","classical") → "E2"`, `("E4","classical") → "E4"` — and a `template=` parameter on `layer8_cpp_simulation`/`run_cpp_brady2d` selecting the table + Lua file pair. Wrap the run so a C++ `std::invalid_argument` (e.g. `alpha[1] < 197/288`) surfaces as `BridgeResult(infeasible=True, reason=<stderr line>)`, not an unhandled failure.
4. **CFL knob:** `run_cpp_brady2d(..., cfl=0.8)` fills the `--{{CFL}}--` marker and must **assert the marker exists in the selected template** when `cfl` is passed (the uncut template hardcodes `cfl = { hyperbolic = 0.8 }` with no marker — a silent `str.replace` no-op would run the wrong CFL). Expose a `cfl_values` list in the layer-8 wrapper for finalist validation (sequential runs; each is 0.1–6 s).
5. **Calibration:** E2_1 α=0, cut template, N ∈ {31, 61}, CFL 0.8 — expect `growth_rate ≈ 0` (|rate| < 0.02 observed → set rate_tol = 2× observed noise floor; record); Gaussian ε=0.1 on the *uncut* template — expect positive growth_rate (the known-unstable anchor from `known_values.json`); E4_1 with α₁ = 0.16 on the cut template — expect `infeasible=True`. Append all numbers to the Findings log.

**Run:** `pytest tests/test_cpp_bridge.py -x -q` (bridge tests must skip cleanly when `build/src/app/shoccs` is absent — follow the existing skip pattern).

### 4.5 End-to-end acceptance (findings item)

For E2_1 α=0 and one deliberately perturbed α (pick after 1.4 — an α the Python margins call marginal): compare (i) `kreiss_margin_over_psi`, (ii) `max_reflection`/round-trip, (iii) `rk4_containment_over_psi`, (iv) cut-template L8 growth_rate at N ∈ {31, 61}, CFL ∈ {0.8, 1.0}. Acceptance: no Python-pass/C++-fail disagreement; a Python-fail/C++-pass is acceptable (proxies may be conservative) but must be logged. Disagreements do not block merge of prior phases (they are the scientific output) — they block only the *default-on* wiring in plan 49.

---

## Risks

| Risk | Mitigation |
|---|---|
| 0.3 recalibration flips known verdicts (tension σ=3.0, Gaussian ε=0.1) | Expected; cross-check against L3 eigenvalues; update tests with citations; Findings log |
| Augmented-determinant construction subtly wrong | Worked upwind example with hand-derived M(s); m=p reduction identity; reviewer agent re-derives independently |
| Right-end mirror sign error | Caught and corrected during plan review (pure reversal, NOT revneg); 2.3's analytic-toy test (κ' = 1+s) locks it in |
| 1985 Lemma 1 operator ordering wrong | Original draft ordering was wrong; corrected in 3.3. p=1 tests cannot detect ordering bugs — the mandatory p=2 cross-validation tripwire in 3.3 can; reviewer still verifies against the paper |
| `scalar_wave` lacks embedded-shape support | 4.3 investigation gate; heat fallback documented as parabolic-only validation |
| E4_1 symbolic cost bleeds into CI | All E4_1 derivation tests `@pytest.mark.slow`; evaluator cache module-scoped in fixtures |
| `SchemeParams` not hashable for lru_cache | Key on the field tuple; add a unit test |
| Conservative closures' structural s=0 zero misread as violation after 0.4 | Perturbation classification at near-origin candidates; Godunov–Ryabenkii regression kept |

## Out of scope (plan 49 candidates)

Optimizer wiring (bounds, denominator-positivity constraints via Sturm sequences, ψ-aggregated objectives, fidelity-ladder rebuild with the cut L8 rung); **fixing the plan-43 category error**: `optimizer.py:61-74` and the `cpp_cutcell_violates_197_288` flag compare the optimizer's uniform-E4 boundary alphas (BL space, e.g. α₁ ≈ 0.162) against E4_1's w₄ ≥ 197/288 quadrature-weight floor — different parameter spaces that share only the name `alpha[1]`; the "analytical and cut-cell feasible regions do not overlap" conclusion is unfounded as stated, the comment/flag must be corrected or retired, and E4-classical optimization needs its own definition in E4_1's true (alpha_2, w₄) coordinates; 2D transverse-wavenumber sweep; staircase/Floquet; corners; parabolic ν=2 program; polyE2_1 interp-row composition; retiring/freezing legacy sweeps.

## Findings log

*(Appended by calibration items 0.3, 1.4, 2.5, 4.3, 4.4.5, 4.5 during execution.)*

### 2026-07-13 — Item 1.2 first attempt blocked; four gks_kreiss.py substrate defects diagnosed (→ item 0.5)

The 1.2 wrapper was implemented per spec and cross-validated: **dense eigenvalues of the assembled E2_1 α=0 half-line operator at N ∈ {20,40,80,160} confirm stability at all five test ψ (max Re λ ≤ 8e-6)** — plan-35's conclusion survives a direct check — while `kreiss_stability_check` returned false-unstable at 4 of 5 ψ via four distinct mechanisms: (1) σ_min(M)→0 as |s|→∞ for augmented m>p matrices (σ_min·|s| → 1/4) while det→∞; (2) unbounded NM witness runaway to |s|~1e155 with σ underflow; (3) near-origin count_mismatch anomalies (E2 sub-sonic axis, both roots unimodular) triaged through the zero-discrimination `_classify_imag_axis` → false outgoing at ψ=0.9; (4) an uncovered second structural zero — the Dirichlet tail-constant κ=1 double zero at s=0 (Σ_{j≥m−p} w_ij ≡ 0 in ψ), null vector (0_nodal, e_{κ→1}) — tripping sigma_tol at ψ=0.9999 via the ψ→1 row-coincidence degeneracy. ψ=0.5 passed only by accident (σ at the 1e-3 barrier 8.8e-8, 9× above tol, same s² zero). Fixes specified as item 0.5; the wrapper itself verified end-to-end and reverted pending 0.5.

### 0.3 (2026-07-13) — All differing closure rows fed to L2: NO verdicts flipped

`_extract_stencil_data` now extracts every left-end row of D that differs from the shifted interior row (m rows, `row_indices = 0..m−1`) and `layer2_kreiss_gks` dispatches to the augmented Kreiss matrix. Recalibration of the three flagged cases (E4, n=20, grid `s_max=10, n_radial=30, n_imag=80, imag_max=15`):

| Case | rows fed (was → now) | GKS verdict (was → now) | witness σ_min (now) | L3 cross-check max Re eig(−D_bc), N∈{20,40,80} |
|---|---|---|---|---|
| tension σ=3.0 | 2 → 4 | stable → **stable** (no flip) | 1.92e-5 at s ≈ 0.0092 − 0.0996j | −4.9e-4, −6.3e-5, −7.8e-6 (all < 0, consistent) |
| Gaussian ε=0.1 | 2 → 4 | stable → **stable** (no flip) | 3.95e-7 at s ≈ 1.2e-6 + 0.1j | +9.83e-2, +9.83e-2, +9.83e-2 (eigenvalue-unstable, still L3's catch) |
| heuristic-vs-rigorous complementarity | n/a (test uses `gks_group_velocity_check` on D directly, no row extraction) | unchanged | — | — |

Notes: (a) both witnesses sit at |s| ≈ 0.1, i.e. pinned against the `min_s_magnitude` mask — the near-origin σ_min dip is the structural consistency zero that item 0.4 will treat explicitly; the Gaussian all-rows σ_min (3.95e-7) is ~3× smaller than the first-p value (1.06e-6) but still above `sigma_tol=1e-8`. (b) classical E4 at the known-good α = [−0.77333…, 0.16240…] extracts m=3 (block row 3 equals the shifted interior row within 1e-10) and stays GKS-stable; L3 confirms (−1.9e-2, −1.9e-3, −1.8e-4). (c) E2 (tension σ=0) extracts m=3 vs p=1 and stays GKS-stable (σ_min 1.28e-3); L3 confirms (−1.9e-2, −1.2e-3, −1.1e-4).

### 1.4 (2026-07-13) — Calibration/control: E2_1 clean; E4_1 legacy closure GKS-unstable across ψ at ALL tested w₄ (incl. shipped default); floor 197/288 verified from the degree-6 denominator (the commit-15c0ddd cubic is the wrong record); 3 witness-search misses explained → acceptance PASS

Script: `scripts/stencil_gen/calibration_plan48_1_4.py` (standalone findings script, not a test; one full run ≈ 6 min, dominated by the 330 s Mathematica-path E4_1 derivation used only for the part-(d) contrast). Settings: the 0.3/0.5/1.2 calibration s-grid `{s_max=10, n_radial=30, n_imag=81, imag_max=15}`; ψ grid = the item-1.3 default 20-point grid; dense cross-checks at N=40 via the item-1.2 test-local half-line assembly (cut Dirichlet rows left, reflected uniform tension-σ=0 closure right; `u_t = −Du`).

**(a) E2_1 α=0 (plan-35 baseline): stable everywhere — control PASSES.** All 20 grid ψ feasible and `is_stable=True` (`kreiss_margin_over_psi`, 1.7 s); per-ψ witness σ_min ∈ [8.23e-8, 0.434], worst at ψ = 1−1e-4 (the ψ→1 row-coincidence edge, as in item 1.3). Dense N=40: max Re eig(−D) ∈ [−9.1e-15, +4.45e-7] over the grid — verdict agreement at all 20 ψ.

**(b) E4_1 in its OWN legacy (α₂→`alpha_0` = 0.1, w₄→`alpha_1`) coordinates — headline: genuinely GKS-unstable across essentially the whole ψ grid at all three w₄ tested, including the shipped default.** Derivation via `derive_cut_cell_scheme(E4_1, ψ)` (legacy zeros path, conserve=True, 5.4 s; exactly 2 free params, confirming the provenance note's reading of temo.py:3160-3176). Note the item-1.1 evaluator canNOT be used here: `make_cutcell_evaluator(E4_1)` dispatches to `derive_cut_cell_mathematica`, which keeps the four uniform-boundary alphas free — the (α₂, w₄) coordinates exist only on the legacy path, so the script builds `CutCellEvaluator` directly on the legacy `CutCellResult`.

| case (alpha_0 = 0.1) | Kreiss verdict on the 20-ψ grid | infeasible ψ | min witness σ_min |
|---|---|---|---|
| w₄ = 0.60 (below floor) | **unstable 20/20** | none on the grid | 2.5e-20 @ ψ=0.95 |
| w₄ = 197/288 + 0.05 ≈ 0.7340 | **unstable 18/20** (the 2 "stable" are search misses, see (c)) | none | 7.4e-19 @ ψ=0.9280 |
| w₄ = 0.70 (shipped test default, control) | **unstable 19/20** (1 search miss) | none | 1.9e-21 @ ψ=0.999 |

- **Infeasible-vs-unstable for w₄ = 0.60:** flagged *unstable*, NOT infeasible, at every grid ψ. The interior denominator pole sits at ψ\* = 0.047021 (root of the true degree-6 denominator, between grid points 0.03 and 0.05); |D₆| is O(10²–10³) at every grid ψ so the absolute `|den| < 1e-12` guard cannot trip there. Probed directly: AT ψ\* the guard does flag infeasible (|D₆| = 2.3e-13); at ψ\* ± 2e-3 the verdict is unstable (σ ≈ 1.6e-19). Dense near-pole amplification: max Re eig(−D) = 22.8 at ψ = 0.03 (vs 0.61 at ψ = 0.01), N-independent over {20, 40, 80}.
- Witness σ_min values are machine zeros of the equilibrated matrix (1e-17…1e-21) at Re(s) ≈ 0.16–0.74 — true determinant zeros, and the witnesses coincide with left-localized dense eigenvalues (e.g. w₄=0.60, ψ=0.001: witness s = 0.65251 − 0.52907j vs dense eigenpair 0.65251 ± 0.52907j). "Worst ψ" is not meaningful when det M vanishes at nearly every grid ψ; the unstable sets are recorded in the script's JSON output.
- The 197/288 floor is orthogonal to stability — both above-floor cases are unstable — consistent with this item's provenance note (parameterization-singularity constraint, not a stability constraint). The (0.1, 0.7) defaults were chosen in plan 26 for conservation + singularity-freedom; nothing in C++ time-steps a ν=1 E4 cut closure today, so this was undetectable before. Stability-optimizing E4_1 in its true (α₂, w₄) coordinates is exactly the plan-49 item already listed under Out of scope.
- Side observation: a few ψ returned defective-κ conservative fails at s ≈ 1.17601 (the E4 interior characteristic polynomial's double-root branch point; w₄=0.60 ψ ∈ {1e-4, 3e-4}, and ψ=0.01 for the other two cases) — unstable direction, consistent with dense.

**(c) Dense cross-check: 77/80 matched-ψ verdicts agree in direction; the 3 disagreements are fully explained → acceptance PASS, Phase 2 unblocked.** The three Kreiss-stable/dense-unstable points (w₄=197/288+0.05 at ψ=0.05, σ=1.0e-3 and ψ=0.0720, σ=2.3e-4; w₄=0.70 at ψ=0.03, σ=4.5e-4) were probed: each has a left-localized dense eigenpair s\* (0.6960±0.4860j, 0.6919±0.4653j, 0.7376±0.4776j), and `min_singular_value_ex(…, s*, equilibrate=True)` returns σ_min = 5.5e-17 / 2.4e-17 / 3.5e-18 with status "ok" — det M(s\*) = 0, i.e. **the determinant criterion agrees with dense; the 30×81-grid + single-Nelder-Mead witness search missed those zeros** (it refined into a benign near-axis local minimum instead). This is a search-robustness gap, not a criterion defect. Recommended hardening before plan-49 wiring (candidate 0.5-style follow-up): multi-start refinement seeded from several coarse-grid local minima rather than the single global argmin. Right-end contamination ruled out: the mirrored (pure-index-reversal, item-2.3 convention) uniform tension-σ=0 E4 right closure is GKS-stable as a half-line problem (σ_min = 2.05e-2), and 0 of the 60 E4_1 configs have any right-localized unstable dense mode — every unstable dense mode is the cut closure's.

**(d) Denominator floor re-derived independently: the E4_1.cpp degree-6 record is the TRUE denominator; the commit-15c0ddd cubic is not; the 197/288 floor itself is verified correct.**
- Structural denominator factors of the legacy blocks (`CutCellEvaluator.denominators` on the legacy result): Dirichlet {D₆, α₁, ψ−1}; floating {D₆, α₁, ψ, ψ−1} — matching E4_1.cpp:17-27's pole comment exactly (ψ=0 pole floating-only, ψ=1 pole in both), plus a w₄=0 pole (factor α₁) not recorded in the comment.
- The unique α₁-dependent ψ-interior-vanishing factor is **D₆(ψ; w₄) = 144w₄ψ³ + 864w₄ψ² + 1584w₄ψ + 1728w₄ + 12ψ⁶ + 162ψ⁵ + 1464ψ⁴ + 5617ψ³ + 8070ψ² + 1721ψ − 1182** — identical (constant ratio 1) to the E4_1.cpp:22-24 comment and to the generated code's `t35`/`t36` reciprocals.
- The commit-15c0ddd cubic 288w₄ + 648ψ + 90ψ² + 12ψ³ − 197 is NOT a factor of any block entry and does not divide D₆ (sympy remainder 3456w₄² + 2820w₄ + (216w₄−4572)ψ² + (2736w₄+14028)ψ − 3546 ≠ 0); it also predicts the wrong pole location (ψ\* = 0.0369 at w₄=0.60 vs the true 0.0470). The commit message is the wrong record of D itself.
- **The floor survives unchanged:** D₆(0) = 1728w₄ − 1182 = 6·(288w₄ − 197); solving D₆ = 0 for w₄ gives a rational function of ψ whose maximum on [0, 1] is attained at ψ = 0 and equals exactly 197/288 (verified symbolically and on a 201-point sample), and ∂D₆/∂ψ has all-nonnegative coefficients for w₄ ≥ 0, so D₆ is strictly increasing in ψ and D₆ > 0 on (0, 1) ⟺ w₄ ≥ 197/288. E4_1.cpp's constructor guard and comment are correct as implemented; only the commit message's polynomial is wrong.
- Guard-calibration note for plan 49: the absolute `|den| < 1e-12` feasibility test is nearly vacuous against D₆ (natural scale O(10²–10³)) — at w₄=0.60 it trips only within ~1e-16 of ψ\* while coefficients are already grossly amplified at ψ = 0.03. Sturm-sequence positivity constraints (already on the plan-49 list) are the right guard; near-term, a scale-normalized (relative) threshold would widen the infeasibility window sensibly.
- Contrast: the item-1.1 canonical `make_cutcell_evaluator(E4_1)` (Mathematica path, 4 free alphas, 330 s) has entirely different denominators — {66α₀ − 73, one bilinear degree-5-in-ψ polynomial in (α₀αᵢ, αᵢ, ψ)} — with no 197/288 anywhere, confirming the provenance note that the floor is specific to the legacy zeros-path (α₂, w₄) parameterization.

**Acceptance (item 1.4): PASS** — no unexplained contradiction between Kreiss verdicts and dense eigenvalues at matched ψ (E2_1: 20/20 agree; E4_1: 57/60 agree, 3/60 explained as witness-search misses with det M(s\*) = 0 confirmed at the dense eigenvalues). Phase 2 is not blocked. Follow-up recommendations recorded above: (i) multi-start witness refinement, (ii) scale-normalized feasibility threshold (plan 49).

### 2.5 (2026-07-14) — Gaussian ε=0.1 mechanism calibration: expected outcome CONFIRMED (each end finite ‖R‖, product > 1 on the axis); round-trip explanation NOT falsified; control caveat recorded for 3.3

Closure: the documented GKS-passes-but-eigenvalue-unstable case — uniform Gaussian ε=0.1, `build_diff_matrix_rbf(n=20, p=2, q=3, nu=1, nextra=0)`; m=4 closure rows on support columns 0..5 (the same all-differing-rows block item 0.3 feeds), interior = classical E4 to 1e-10 (flat limit); right end = nu=1 codegen revneg image of the left rows (`D[::-1, ::-1] == −D` exactly), so A_R comes from `mirror_closure(..., from_physical_right=False)` (item 2.3 production entry). Grid: `default_reflection_s_grid()` (241-point strip s = 1e-6 + iω, |ω| ≤ 20, near-origin exclusion |ω| ≥ 1e-3 → 240 points). Test added: `tests/test_reflection.py::TestGaussianEps01Mechanism` (fast, ~0.9 s; names contain `gaussian` per the -k rule).

- **Each end individually GKS-stable with finite ‖R‖.** A_L: `finite=True`, no defective points, max ‖A_L‖ = 3.72e7 at s = 1e-6 − 0.1667j — the grid point nearest the excluded origin, i.e. the tail of the benign structural consistency pole at s = 0 compounded by the near det-zero at s ≈ 0.1j recorded in the 0.3 findings (σ_min ≈ 3.95e-7 there). A_R: `finite=True`, max ‖A_R‖ = 8.94e6 at the grid edge s = 1e-6 − 20j (evanescent-mode normalization growth ~ |μ|^j — cancelled in 3.3 by the transit factors K^Δj/Mdiag^−Δj). Corroborating single-boundary Kreiss checks (0.3 calibration grid `{s_max=10, n_radial=30, n_imag=80, imag_max=15}`): left `is_stable=True` (witness at the |s| = 1e-3 barrier, σ = 9.6e-16, benign structural classification — as in 0.4); right `is_stable=True`, witness σ = 2.51e-2 at s ≈ 1e-6 + 1.3722j (the E4 sonic frequency).
- **Two-boundary product: > 1 on essentially the whole sampled axis.** Pointwise ‖A_L(s)‖·‖A_R(s)‖ > 1 at ALL 240 grid points; max = 8.0e13 at the grid edge (evanescent-inflated); restricted to the propagating band |ω| ≤ ω_sonic = 1.3722: max = 1.19e12 at ω = −1/6, 16/16 band points > 1, mid-band anchor ω = 1.0: ‖A_L‖ = 2.55e5, ‖A_R‖ = 4.18e4, product = 1.06e10. Dense anchor unchanged: max Re eig(−D_bc) = +9.83e-2. **The plan's expected outcome holds; the round-trip explanation is not falsified** — the Gaussian case now has a per-end analytic account (finite reflections, unbounded-in-product round trip) instead of only the L3 eigenvalue symptom.
- **Control caveat (recorded for 3.3; not a test):** the same product screen on the GKS-stable AND eigenvalue-stable tension σ=3.0 closure (dense max Re = −4.9e-4) also gives product > 1 everywhere (max 4.3e12 at the edge; band max 1.4e11 at ω = +1/6; A_L max 7.4e6, A_R max 2.1e6, both finite). Consistent with the math — the decoupled product only *upper-bounds* ρ(E_Δj), so product > 1 cannot certify instability, it can only fail to rule the mechanism out (which is all 2.5 claims for the Gaussian). The magnitudes are dominated by the same two artifacts at both closures (near-origin structural-pole tail, evanescent edge growth). Discrimination between the Gaussian (unstable) and tension (stable) pairs is exactly the Δj-phased round-trip radius ρ(E_Δj) of item 3.3 — its acceptance test (Gaussian flagged at the N=20-matching Δj; E2_1 α=0 radius < 1) must rest on ρ(E_Δj), as the plan already specifies, not on the bare product.

### 4.3 (2026-07-14) — L8 cut-cell capability gate: **Outcome A — scalar_wave supports embedded shapes end-to-end; 4.4 proceeds with `system = "scalar wave"`** (gate PASSES); plus three calibration caveats that materially affect 4.4.5

Determination made by code reading **and** by actually running scalar-wave-with-shapes through `simulation_cycle::run()` via the built `build/src/app/shoccs` binary (the reading-only evidence left exactly that gap: the repo's cycle smoke tests use heat + sphere, and `scalar_wave.t.cpp`'s sphere tests stop at the system level). Probe configs and CSVs under the session scratchpad (`p48_43/`); nothing committed but this log entry.

**Evidence — the capability chain (file:line):**
- Dispatch: `src/systems/system.cpp:117-120` routes `type == "scalar wave"` to `scalar_wave::from_lua`; `simulation_cycle::from_lua`/`run()` are system-agnostic over the variant (`src/simulation/simulation_cycle.cpp:31-143`), and the graph-build hook at `:72-75` explicitly covers scalar_wave.
- Shapes ingestion: `src/systems/scalar_wave.cpp:143-172` — `system.center/radius` takes precedence for the exact solution; otherwise center/radius are pulled from the first sphere in `shapes`. Mesh/object geometry and object BCs come from the shared `mesh::from_lua` + `bcs::from_lua` path (`:174-189`); the gradient is built with `object_bcs` (`:55`), i.e. cut operators.
- Object-BC application: constructor zeroes the wave-speed coefficients on object-Dirichlet R subsets (`scalar_wave.cpp:87-95` via `m.dirichlet_object_desc`); `update_boundary` assigns the exact solution at object-Dirichlet Rx/Ry/Rz points (`:348-354`) and is invoked at **every RK stage** (`src/temporal/rk4.cpp:31,46`); stats/write are object-aware via `detail::compute_scalar_stats(m, object_bcs, …)` (`:379-380, :414-416`).
- Existing tests: `src/systems/scalar_wave.t.cpp:43-155` and `:159-273` already exercise scalar_wave + sphere (3D, `boundary_condition = "dirichlet"`, E2) — initialize, stats, update_boundary values on Dirichlet faces, rhs finiteness, graph≡eager. The cycle smoke tests (`src/simulation/simulation_cycle.t.cpp:22-99, 101-176`) use heat + sphere only — that was the unverified link.
- `logs/system.csv` unchanged: scalar_wave's header (`scalar_wave.cpp:98-101`) is byte-identical to heat's (`src/systems/heat.cpp:47-49`) — 15 columns `Timestamp,Time,Step,Linf,Min,Max,Domain_Linf,Domain_ic,Rx_Linf,Rx_ic,Ry_Linf,Ry_ic,Rz_Linf,Rz_ic,Wall_ms` — and `cpp_bridge._parse_system_csv` (cols 1, 3; `cpp_bridge.py:90-108`) parses the cut runs' CSVs unmodified.
- Stale doc, for the record: `docs/CAPABILITY_AUDIT.md:140-144` ("scalar_wave … NOT wired through `simulation_cycle`") predates the L8 bridge — `lua-configs/brady_livescu_4_3.lua` has `system = { type = "scalar wave" }` and the bridge runs it through the full loop; re-verified live below.

**Evidence — end-to-end runs** (Brady–Livescu domain `[0,√2]²`, N=31, RK4; cut sphere center (0.7071, 0.7071, 0), radius 0.25, `boundary_condition = "dirichlet"`; growth_rate = trailing-half least-squares slope of log L∞, i.e. item 4.4.2's metric):

| scheme | shapes | CFL | end state | growth_rate |
|---|---|---|---|---|
| E2_1 α=0 (`order=1, type="E2", alpha={0}`) | none | 0.8 | L∞ = 14.99 at t = 0.490 (max_error abort) | +7.93 |
| E2_1 α=0 | sphere | 0.8 | L∞ = 14.99 at t = 0.490 | +7.93 (identical to uniform) |
| E2_1 α=0 | sphere | 0.4 | L∞ = 11.58 at t = 0.471 | +7.61 |
| E2_1 α=0 | sphere | 0.2 | L∞ = 10.10 at t = 0.462 | +7.55 |
| E4u classical α = [−0.77333…, 0.16240…] | none | 0.8 | L∞ = 1.75e-3 at t = 10.03 | **+0.001** |
| E4u classical α | none | 0.8 | L∞ = 1.28e-3 at t = 1.02 | +0.333 (error transient, see caveat 3) |
| E4u classical α (uniform-only stencil) | sphere | 0.8 | L∞ = 12.46 at t = 0.754 | +6.63 (silent, exit 0) |

The cut run exits 0, logs `found sphere of radius 0.25 and center 0.7071, 0.7071, 0`, generates 44 cut points with realized ψ ∈ [0.0881, 0.7920] (`logs/geometry.csv`), and writes the standard system.csv. Rx/Ry error columns are exactly 0 — consistent with exact Dirichlet enforcement at object points every stage (values assigned from the exact solution, error evaluated against the same solution). The end-to-end machinery — sphere → cut geometry → ψ-parameterized closures → CSR wall coupling → per-stage object BC → stats/CSV — works. **Gate: PASS (Outcome A). Item 4.4 proceeds with `system = "scalar wave"` on the cut template.**

**Caveat 1 (materially affects 4.4.5): E2_1 α=0 is fully-discretely UNSTABLE on this template — uniform and cut identically.** Blow-up is at fixed *physical* time (t ≈ 0.46–0.49 across CFL ∈ {0.2, 0.4, 0.8}; rate ≈ +7.5…+7.9 per unit time), i.e. a semi-discrete mechanism, not an RK4-region violation; and the uniform (shapes = {}) trace is essentially identical to the cut trace (growth_rate matches to 3 digits), i.e. **not shape-induced**. This does not contradict the Phase 0–3 certification: the Python path certifies the *Dirichlet* half-line closure (items 1.2/1.4: stable at all ψ, dense-eigenvalue cross-checked), while this 2D template's xmax/ymax walls are *unspecified* → default **floating** (`src/operators/boundaries.cpp:44`), so E2_1's `nbs_floating` closure at α=0 (`E2_1.cpp:124`, all four alphas zero-padded per `E2_1.cpp:19-21`) governs the outflow ends — exactly the block the plan's constraints exclude from analysis ("the floating block is never analyzed for stability"). Candidate mechanism, not established here (an all-Dirichlet-walls probe blows up faster, t = 0.23, but Dirichlet at outflow is over-specified, so that probe is uninformative). Consequence for 4.4.5: the expectation "E2_1 α=0, cut template, CFL 0.8 → growth_rate ≈ 0, |rate| < 0.02" is **contradicted by today's uniform baseline** (+7.9). 4.4 must either (a) choose template BCs so certified closures govern every boundary (e.g. Dirichlet on the inflow walls is fine — the issue is the floating outflow closure), (b) recalibrate the E2_1 α=0 expectation to the measured *uniform* baseline and treat the cut-vs-uniform **delta** as the cut-cell signal (measured delta today: ≈ 0), or (c) analyze/replace the α=0 floating closure. Filed as follow-up work alongside plan-49 wiring: GKS analysis of the floating uniform closure is currently a certification hole for any template with default-floating walls.

**Caveat 2 (justifies 4.4.3's separate dispatch table): E4u + shapes is a *silent* trap, not a loud one.** The uniform-only `E4u` stencil accepts the sphere without any exception (exit 0, no stderr), produces a garbage cut discretization, and blows up at rate +6.63. The 4.4.3 infeasible-wrap (catching `std::invalid_argument`) will NOT catch a wrong-table dispatch — only the separate `_L8_CUT_SCHEME_TYPE` table mapping exclusively to cut-capable types (`"E2"`, `"E4"`) prevents it. Recommend 4.4's tests assert that the cut template refuses (or at least never silently accepts) uniform-only scheme types.

**Caveat 3 (rate_tol calibration): the growth-rate metric needs the standard t_final = 10 window.** The stable E4u anchor reads +0.001 over t_final = 10 but +0.333 over t_final = 1 — the trailing half of a short window sits inside the error transient toward the ~1.7e-3 plateau, far above the proposed rate_tol = 0.05. 4.4.5's noise-floor calibration must use t_final = 10 (the template default).

**Side observation:** the in-C++ eigenvalue cross-check rung is unavailable: `system = "eigenvalues"` (`hyperbolic_eigenvalues`) segfaults through `simulation_cycle::run()` (exit 139, both uniform and cut, N=31 E2_1) — consistent with `docs/CAPABILITY_AUDIT.md:145-149` ("diagnostic, not an integrator … never time-stepped"). Not needed for 4.4; noted so nobody reaches for it during 4.5.

### 4.4.5 (2026-07-14) — L8 cut-rung calibration: infeasible-wrap and unstable-anchor expectations CONFIRMED; rate_tol kept at 0.05 (38× above the measured noise floor); the plan's "E2_1 cut growth_rate ≈ 0" expectation — already contradicted by 4.3 caveat 1 — recalibrated per option (b), and the cut-vs-uniform DELTA is itself nonzero once the sphere realizes small ψ (new finding, materially affects 4.5)

Script: `scripts/stencil_gen/calibration_plan48_4_4.py` (standalone findings script, committed like 1.4's; full run ≈ 1 s of C++ across 19 shoccs invocations). Template sphere (item 4.4.1): center (0.8441, 0.6177), radius 0.2422, chosen so the realized ψ set spans the extremes at EACH N individually — N=31: 44 cut points, ψ ∈ [1.57e-3, 0.99927]; N=61: 84 points, ψ ∈ [5.70e-6, 0.99853] (< 0.05 and > 0.95 asserted per N in `tests/test_cpp_bridge.py`, stronger than the plan's across-N requirement). The Python ray-cast replication of `object_geometry.cpp:70-95` matches the binary's `logs/geometry.csv` to ~1e-12 (same 44 points, same min/max). `growth_rate` = trailing-half least-squares slope of log L∞ (item 4.4.2); t_final = 10; `max_error = 10` aborts unstable runs early (exit 0, trace intact), so rates on unstable runs are blow-up-window steepnesses, not asymptotic modal rates — verdict-grade only.

| config | template | N | CFL | growth_rate | final L∞ | t_end |
|---|---|---|---|---|---|---|
| E2_1 α=0 | cut | 31 | 0.8 | **+16.33** | 18.9 | 0.377 |
| E2_1 α=0 | cut | 61 | 0.8 | **+31.84** | 14.8 | 0.226 |
| E2_1 α=0 | uniform | 31 | (0.8 hardcoded) | +7.93 | 15.0 | 0.490 |
| E2_1 α=0 | uniform | 61 | (0.8 hardcoded) | +9.53 | 10.8 | 0.377 |
| Gaussian ε=0.1 | uniform | 31 | (0.8) | **+1.33** | 11.0 | 9.77 |
| E4u classical α | uniform | 31 | (0.8) | +0.0013 | 1.75e-3 | 10.03 |
| E4u classical α | uniform | 61 | (0.8) | +0.0013 | 1.19e-4 | 10.01 |
| E4_1 (α₀, w₄) = (0.1, 0.16) | cut | 31 | 0.8 | — | — | **infeasible** |

- **(d) infeasible wrap confirmed:** the E4_1 constructor's `std::invalid_argument` (SIGABRT, exit −6) surfaces as `BridgeResult(infeasible=True, reason="what():  E4_1: alpha[1] must be >= 197/288 ≈ 0.684 …")` — a parameterization failure, never a stability verdict. **(c) unstable anchor confirmed:** Gaussian ε=0.1 on the uncut template gives +1.33 (blow-up near the end of the window, t = 9.77).
- **rate_tol = 0.05 kept (default).** The plan's 2×-noise-floor formula presumed a stable cut E2_1 run (contradicted, see below), so the noise floor anchors on the stable uniform E4u rung: +0.0013, N-independent over {31, 61}, on the standard t_final = 10 window. 0.05 sits ~38× above that floor and ~26× below the smallest measured unstable rate (+1.33); every measured verdict is separated by two orders of magnitude on each side. 4.3 caveat 3 reconfirmed: t_final = 1 windows sit inside the error transient (+0.33 for the stable anchor) — rate verdicts require the t_final = 10 default.
- **(a)/(b) headline — the cut-vs-uniform delta is NOT zero for this item's ψ-spanning sphere:** delta = +8.40 at N=31, +22.31 at N=61 (4.3's probe sphere, ψ ∈ [0.088, 0.79] at N=31, had delta ≈ 0). CFL sweep (4.4.4 knob; CFL ∈ {0.1, 0.2, 0.4, 0.8}): rates CFL-independent (N=31: 15.4–16.3 over an 8× dt range; N=61: 31.8–32.0) → **semi-discrete mechanism, not an RK4-region/cut-CFL violation**. Rate ≈ doubles as h halves (16 → 32) → fixed Re(s) > 0 in the scaled variable s = λh, i.e. the extra growth lives in the near-boundary/cut layer, consistent with a closure-scale mode, not a physical-domain mode.
- **(g) control sphere discriminates:** the 4.3 mild-ψ sphere ((0.7071, 0.7071), r = 0.25) through the SAME pipeline reproduces 4.3 exactly at N=31 (+7.929, t_end 0.4903 — delta ≈ 0, so the new code path is consistent), but at N=61 (realized ψ ∈ [0.031, 0.83]) it too goes fast: +37.8, delta +28.3. Correlation across the four (sphere, N) pairs: delta ≈ 0 only when min realized ψ = 0.088; delta large for min ψ ∈ {5.7e-6, 1.6e-3, 0.031}; near-1 ψ not implicated (the mild sphere's max ψ is 0.83 yet shows the largest delta). **Small realized ψ (≲ 0.03–0.09 threshold) is the prime suspect.** Note `E2_1.cpp` applies NO ψ clamp (only `E4_1.cpp:101,235` clamps to [1e-4, 1−1e-4]), so N=61 evaluates the closure at ψ = 5.7e-6 raw.
- **Attribution NOT established here** (findings, not analysis): the uniform baseline is itself unstable (default-floating xmax/ymax walls — 4.3 caveat 1, certification hole already filed), and the 1D Dirichlet half-line certification (1.2/1.4: E2_1 α=0 stable at all ψ incl. 1e-4, dense-verified) covers neither 2D transverse coupling, nor the varying-coefficient radial flow, nor outflow-side Dirichlet over-specification around the sphere. Discriminating probes for 4.5: sphere variants realizing only-small vs only-large ψ; N=41 (fresh ψ draw); an all-certified-boundary configuration once the floating-wall hole is fixed; per-ψ-threshold bisection of the delta.
- **Consequence for 4.5 acceptance:** E2_1 α=0 is a Python-pass (half-line Dirichlet block) / C++-fail (this template) pair. Under the plan's rule this disagreement is logged, but it is not yet attributable to the certified object: the template's floating walls dominate the baseline, and the cut delta correlates with small ψ where the *2D embedded* problem departs from the certified 1D half-line one. 4.5 must work with the delta-vs-uniform signal (4.3 option (b)) plus the discriminating probes above, not the raw rate.
