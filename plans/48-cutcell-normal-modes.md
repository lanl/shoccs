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
