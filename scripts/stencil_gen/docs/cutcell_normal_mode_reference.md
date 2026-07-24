# Cut-Cell Normal-Mode Stability Reference (plan 48, Phases 0–3)

Theorem-backed stability analysis of ψ-parameterized cut-cell closures —
the Trefethen program (1983 determinant / 1984 reflection coefficients /
1985 two-boundary round trip) applied to the closures the C++ solver
actually time-steps — exposed as smooth, worst-cased-over-ψ margins.

Modules: `stencil_gen/gks_kreiss.py` (substrate), `stencil_gen/cutcell_eval.py`
(item 1.1), `stencil_gen/cutcell_normal_modes.py` (1.2/1.3),
`stencil_gen/reflection.py` (2.1–2.4, 3.3), `stencil_gen/interval_operator.py`
(3.1/3.2), report entry point in `stencil_gen/brady2d_stability.py` (2.4/3.4).

## Conventions (pinned; sign errors are the #1 historical pain point)

- **Semi-discrete sign convention:** everything analyzes
  `u_t = −Σ_k a_k u_{n+k}` (the `kappa_roots` docstring convention).
  Raw derivative-operator rows are passed through **unflipped**; interval
  operators return the raw D and stability is read off `eig(−D)`.
- **The Dirichlet (wall-eliminated) block is the canonical analysis
  object** (plan 48 hard constraint). `derivative.cpp` strips the wall
  column into the CSR B matrix; under homogeneous Dirichlet the wall
  value is 0, so dropping the column is exact. The floating block is
  never analyzed for stability.
- **No time integration in Python** (hard constraint): all quantities are
  resolvent-based (σ_min evaluations) or algebraic (root loci). No matrix
  powers, no `expm`, no stepping. RK4 reasoning is Phase 4 (resolvent
  containment in Python; actual stepping only in C++).
- Unit grid spacing h = 1 throughout (matching every repo builder).
- ψ ∈ (0, 1]; ψ = 1.0 means a **uniform domain boundary** at that end
  (the C++ convention of passing psi = 1.0 to `nbs`).

## Math

### Augmented (generalized) Kreiss matrix — item 0.2

Left half-line, closure rows i = 0..m−1 at grid nodes n_i (default
n_i = i), centered interior with p = L_left = −min(offsets) admissible
roots κ_ℓ(s) (|κ| < 1 for Re(s) > 0). The interior recurrence constrains
φ only on n ≥ m−p, so the general decaying solution is

```
φ_n = Σ_{ℓ=1..p} α_ℓ κ_ℓ^n   for n ≥ m−p,
φ_0..φ_{m−p−1}                free nodal unknowns.
```

Unknowns x = (φ_0, …, φ_{m−p−1}, α_1, …, α_p) ∈ C^m; row i of M(s):

- nodal column q (q < m−p): `M[i,q] = s·δ_{n_i,q} + w_{i,q}`
  (temporal δ against the row's grid node n_i, not the row index);
- root column ℓ: `M[i, m−p+ℓ] = s·κ_ℓ^{n_i}·1[n_i ≥ m−p]
  + Σ_{j ≥ m−p, j ∈ support(i)} w_{i,j}·κ_ℓ^j`.

GKS eigensolution ⟺ det M(s) = 0 for some Re(s) ≥ 0. Reduces to the
classical square Kreiss matrix when m = p. Every real closure has m > p
(cut-cell Dirichlet blocks: m = R−1 = 3 vs p = 1 for E2_1; 4 vs 2 for
E4_1) — the augmented matrix is what makes the check non-vacuous.

### Witness-search robustness — item 0.5

For augmented m > p matrices, σ_min(M(s)) → 0 as |s| → ∞ while
det M(s) → ∞, so the raw σ_min proxy is meaningless at large |s|:

- **Column equilibration** (unit ∞-norm columns) makes σ_min a relative
  margin at all |s| (det-zero locations invariant).
- **Eigensolution bound**: any GKS eigensolution satisfies
  |s| ≤ S_max = max over rows of Σ_j |w_ij| (sup-norm/Gershgorin);
  the grid and the Nelder-Mead refinement are confined to it.
- **Structural benign zeros at s = 0** are discriminated *structurally*,
  never via the perturbation classifier: the conservative full-constant
  zero (right singular vector aligned with x_const = (1,…,1, e_{κ→1}),
  row sums ≈ 0) and the Dirichlet tail-constant zero (x_tail =
  (0_nodal, e_{κ→1}), tail sums Σ_{j≥m−p} w_ij ≈ 0 — identically in ψ
  for wall-eliminated blocks, a *double* zero). The vanishing order
  ν ∈ {1, 2} is fitted along a ray into the RHP and the deflated margin
  σ_min(s)/|s|^ν reported.

### Feasibility guard — item 1.2

Before any assembly, the closure's structural denominator polynomials
(`CutCellEvaluator.denominators`) are evaluated at (ψ, α); if any
|denominator| < 1e-12 the check returns the distinct `CutCellInfeasible`
marker (no `is_stable` attribute). Denominator vanishing is a
**parameterization failure** — the symbolic solve degenerates — and is
never conflated with GKS instability.

### Reflection coefficients — items 2.1/2.2

Mode partition at s (strip s = iω + 1e-6): rightgoing |κ| < 1,
leftgoing |μ| > 1, borderline unit-modulus roots classified by the
shared perturbation helper; centered 2p+1 interiors have exactly p of
each for Re(s) > 0. With a unit-amplitude incident leftgoing mode μ_k,

```
φ_n = μ_k^n + Σ_ℓ α_ℓ κ_ℓ^n  (n ≥ m−p),   φ_q free (q < m−p),
```

the closure equations read `M(s)·x = −b_k(s)` with
`b_k[i] = s·μ_k^{n_i}·1[n_i ≥ m−p] + Σ_{j ≥ m−p} w_{i,j}·μ_k^j`.
The reflection matrix R(s) is p×p (column k = α-block of x);
`max_reflection = max_s ‖R(s)‖₂` over the imaginary-axis strip grid.
R is invariant under joint row scaling of [M | b] (σ_min is not) and
blows up exactly at det M(s₀) = 0 — the properties that make it a
better-conditioned margin than σ_min.

### Right-end mirror — item 2.3

The coordinate mirror n → N−n′ is **pure index reversal, no negation**
(interior a_k → a_{−k}, κ → 1/κ). The negation belongs to *codegen's
construction* of physical right-wall rows from stored left-wall rows
(ν = 1 revneg: `D[N−1−i, N−1−j] = −B[i, j]`). `mirror_closure(...,
from_physical_right=...)` covers both entry points; the production
cut-cell path uses `from_physical_right=False` (stored left rows +
revneg symmetry), which is exactly the right block
`build_interval_operator` assembles.

### Two-boundary round trip — item 3.3 (Trefethen 1985 Lemma 1)

At s on the axis grid, with K = diag(κ) (rightgoing), Mdiag = diag(μ)
(leftgoing), and per-end reflection matrices A_L(s; ψ_L), A_R(s; ψ_R)
re-indexed onto one mode basis via the exact reciprocal-root
correspondence of the mirror:

```
E_Δj(s) = Mdiag^{−Δj} · A_R(s; ψ_R) · K^{Δj} · A_L(s; ψ_L)
```

(right-to-left: reflect at the left end, transit right, reflect at the
right end, return). GKS eigensolution of the pair ⟺ some eigenvalue of
E_Δj equals 1. Both reflection matrices are referenced at their end's
node (0 and Δj); Δj = N − 1 for an N-node interval, and
`Δj_min = max(t_L + t_R − 1, max_node_L + max_node_R + 1)` (supports and
rows may not overlap — stricter than the 3.1 assembly minimum).
ρ(E_Δj) is **not monotone in Δj** (pure-phase transit ⇒ Δj-specific
resonances), so `round_trip_radius = max over the s-grid AND over
Δj ∈ {Δj_min, …, Δj_min + 5} of ρ(E_Δj(s))`. The decoupled screen
`reflection_product = max_s ‖A_L‖·‖A_R‖` upper-bounds ρ on the axis
(‖K‖ = ‖Mdiag^{−1}‖ = 1 there): product < 1 certifies all Δj at once;
**product > 1 alone cannot flag** (the tension-σ=3.0 control also has
product > 1 — see calibration).

Caveats (both enforced by the code's routing):

- radius < 1 is a certificate **only** given per-end half-line
  stability (Lemma 1's hypothesis) — a per-end det-zero on the grid sets
  ρ = inf; off-axis per-end poles (e.g. the E2_1 outflow eigenvalue at
  real s) are the per-end/interval tiers' job, not the axis radius's.
- Defective mode bases (sonic points, e.g. E2's s = ±i) and failed
  reciprocal matching route through `defective_points` (ρ = nan,
  `finite=False`), never as clean values.

### Interval operator and resolvent scan — items 3.1/3.2

`build_interval_operator` mirrors `derivative.cpp`'s
`[dense_left | circulant | dense_right]` per-line layout with wall
columns dropped and the right block produced by the 2.3 mirror
convention. Minimum length N = m_L + m_R (closure rows may not overlap;
at equality the line is all-closure — exactly the dangerous case — and
still assembles); column supports must fit (max(t_L, t_R) ≤ N);
violations raise `IntervalTooShort`. The scan evaluates, per
(ψ_L, ψ_R, N): dense `max Re eig(−D)` and a grid-sampled Kreiss
constant on an explicit refined near-axis grid — resolvent quantities
only.

## API

| Function / class | Module | Purpose |
|---|---|---|
| `make_cutcell_evaluator(scheme, bc="dirichlet")` | `cutcell_eval` | Cached (lru) lambdified evaluator; `.rows(psi, alpha)` → wall-eliminated Dirichlet rows + `row_indices`, `.interior()`, `.denominators`, `.dims` |
| `kreiss_matrix_general(iw, io, brows, s, *, row_indices)` | `gks_kreiss` | Augmented m×m Kreiss matrix (item 0.2) |
| `min_singular_value_ex(..., equilibrate=True)` | `gks_kreiss` | (σ, status) with status ∈ {ok, defective, count_mismatch} |
| `kreiss_stability_check(iw, io, brows, *, row_indices)` | `gks_kreiss` | Rigorous half-line GKS verdict (`KreissResult`) with 0.4/0.5 near-origin + robustness treatment |
| `kreiss_cutcell_check(scheme, alpha, psi)` | `cutcell_normal_modes` | ψ-parameterized check; `CutCellInfeasible` on denominator vanishing |
| `kreiss_margin_over_psi(scheme, alpha, psi_grid)` | `cutcell_normal_modes` | Worst-case-over-ψ margin (`CutCellPsiSweep`); default 20-point grid clustered near 0, matched to the C++ clamp [1e-4, 1−1e-4] |
| `mode_split(iw, io, s)` | `reflection` | Rightgoing/leftgoing/borderline partition (`ModeSplit`) |
| `reflection_matrix(iw, io, brows, s, *, row_indices)` | `reflection` | p×p R(s) |
| `max_reflection(iw, io, brows, *, row_indices, s_grid)` | `reflection` | `ReflectionResult` (max_norm, argmax_s, finite) over the strip grid |
| `mirror_closure(brows, interior, *, from_physical_right, row_indices)` | `reflection` | Item-2.3 mirror (both entry points; production uses `False`) |
| `reflection_over_psi(scheme, alpha, psi_grid, side)` | `reflection` | ψ-swept reflection margin (`CutCellReflectionSweep`) |
| `min_round_trip_delta_j(left_rows, mirrored)` | `reflection` | Δj_min (ties to `IntervalTooShort`) |
| `round_trip_factors / round_trip_matrix(..., s, delta_j)` | `reflection` | Matched A_L, A_R, K, Mdiag; composed E_Δj(s) |
| `round_trip_check(iw, io, left_rows, mirrored, ...)` | `reflection` | `RoundTripResult`: radius, product screen, defective routing |
| `cutcell_round_trip(scheme, alpha, psi_L, psi_R)` | `reflection` | Cut-cell pair entry point (feasibility-guarded) |
| `build_interval_operator(scheme, alpha, psi_L, psi_R, N)` | `interval_operator` | N×N two-ended interval operator D |
| `interval_stability_scan(scheme, alpha, *, psi_pairs, N_values)` | `interval_operator` | Dense abscissa + Kreiss-constant scan (`IntervalScanResult`) |
| `default_interval_psi_pairs(psi_grid)` | `interval_operator` | ~50-pair thinning: corners + diagonal + fixed-seed Latin sample |
| `cutcell_stability_report(scheme, alpha, *, psi_grid, psi_pairs, N_values, n_delta_j)` | `brady2d_stability` | Opt-in report; fills `StabilityReport.layer_cutcell` |

### `layer_cutcell` report keys (items 2.4 + 3.4)

Populated **only** by `cutcell_stability_report` — never by the L1–L8
cascade; `extract_field` reaches them via dotted paths
(`"layer_cutcell.max_reflection"`). Objective/alias wiring is plan 49.

| Key | Source | Meaning |
|---|---|---|
| `worst_kreiss_sigma` / `worst_psi` | item 1.3 | min over feasible ψ of the witness σ_min (and its argmin) |
| `max_reflection` / `argmax_psi` | item 2.4 | max over feasible ψ of the axis reflection norm |
| `reflection_product` | item 3.3 | max over feasible (ψ_L, ψ_R) pairs of `max_s ‖A_L‖·‖A_R‖` |
| `round_trip_radius` | item 3.3 | max over pairs, s-grid, and the Δj window of ρ(E_Δj(s)) |
| `interval_worst_abscissa` | item 3.2 | max over pairs and the N ladder of `max Re eig(−D)` |
| `infeasible_psi` | item 1.2 | sorted union of ψ (grid points and pair ends) with vanishing structural denominators |

## Calibration results (plan 48 Findings log, condensed)

- **Item 0.3 — feeding ALL differing closure rows flipped no verdicts**
  (tension σ=3.0 and Gaussian ε=0.1 both stay half-line GKS-stable with
  the full 4-row block; L3 dense eigenvalues concur in direction).
- **E2_1 α=0 (plan-35 baseline): stable everywhere.** All 20 default
  grid ψ feasible and Kreiss-stable (witness σ_min ∈ [8.2e-8, 0.43],
  worst at the ψ→1 row-coincidence edge); dense N=40 cross-check agrees
  at all 20 ψ (max Re eig(−D) ≤ 4.5e-7).
- **Historical (superseded by e4-align): the pre-fix E4_1 closure was a
  one-node-downwind mis-port of Brady & Livescu (2021) and was genuinely
  GKS-unstable across essentially the whole ψ grid**, with machine-zero
  det M witnesses that coincide with left-localized dense eigenvalues.
  This finding drove the e4-align rewrite, which replaced the mis-ported
  closure with the published 4-parameter cut-cell family (see
  `docs/reference/py-derivation.md`); the legacy (α₂, w₄) parameterization
  and its denominator-positivity floor no longer exist. Full derivation
  detail lives in `cutcell_trefethen_theory.md` and
  `e4_1_determinant_analysis.md`.
- **Gaussian ε=0.1 (the documented GKS-passes-but-eigenvalue-unstable
  case) now has an analytic mechanism.** Full 4-row block: each end
  individually half-line stable with finite ‖R‖, two-boundary product
  > 1 across the sampled axis (item 2.5). Sharpened at item 3.3: the
  operator L3 actually eigendecomposes is the wall-eliminated
  D[1:, 1:], whose 3-row left closure is itself half-line **unstable**
  (witness s ≈ 0.0983 + 1.378j = the N→∞ limit of the dense
  eigenvalue), and the round-trip identity ρ(E_18(s*)) = 1 holds to
  ~4e-12 at the N=20 dense eigenvalue. The axis radius stays < 1
  (0.8706) because the per-end pole is off-axis — the flag comes from
  the product screen, per the plan's acceptance criterion.
- **Control caveat:** the stable tension-σ=3.0 pair also has product
  > 1 (near-origin structural-pole tail + evanescent edge growth), so
  the bare product is a screen, not a verdict; discrimination is the
  Δj-phased radius and, for off-axis per-end poles, the per-end and
  interval tiers.
- **E2_1 outflow law (item 3.2 finding):** under `u_t = −D u` the cut
  Dirichlet closure at the RIGHT (outflow) end carries a right-localized
  eigenvalue at exactly `(1 − ψ_R)/(1 + ψ_R)` — N-independent, present
  for every ψ_R < 1 — via a per-end det-zero at that real s, while the
  cut closure at the LEFT (inflow) end is stable. The interval scan and
  the per-end check own this eigenvalue; the axis round-trip radius
  correctly does not flag it. This is why `interval_worst_abscissa` is
  large for any pair set containing cut right ends. **Note:** Dirichlet
  at an outflow end is ill-posed at the continuous level (it imposes
  data against the outgoing characteristic), so this mode is the
  expected symptom of over-specification, not a closure defect, and
  stabilizing it is not a design goal. Such configurations appear in the
  test suite only as exact validation anchors; the correct leeward
  treatment for advection is the outflow (floating) closure.
- **E2_1 α=0 round trip:** radius < 1 at all tested dense-stable pairs
  (approaches 1 − 6e-6 from below — near-origin consistency physics).

## Limits

- **1D**, single straight cut per end: no transverse wavenumber (2D), no
  staircase/Floquet, no corners (all later plans).
- **Semi-discrete only**: these are GKS/resolvent statements about
  `u_t = −D u`. The fully-discrete bridge (RK4 stability-region
  containment, Kreiss–Wu / Reddy–Trefethen) is plan 48 Phase 4; RK4 is
  only ever *run* in C++ (L8).
- **Homogeneous Dirichlet wall, ν = 1 (advection)** closures; the
  parabolic ν = 2 program is out of scope.
- Kreiss verdicts are half-line statements; fixed-Δj two-boundary
  resonances need the 3.2/3.3 tiers, and marginal (generalized,
  z ≈ 1) eigensolutions are handled by the structural near-origin
  discrimination, not excised.
- The witness search is grid + single refinement; item-1.4 recorded 3/60
  E4_1 misses (det M(s*) = 0 confirmed at the dense eigenvalues) —
  multi-start refinement is a recommended plan-49 hardening.
- Optimizer wiring (objectives, aliases, bounds, Sturm-sequence
  denominator constraints) is deliberately **not** done here (plan 49);
  `layer_cutcell` is opt-in and never populated by the cascade.

## Tests

- `tests/test_gks_kreiss.py` — substrate (0.1–0.5): augmented matrix,
  equilibration, S_max bound, structural zeros, worked upwind example.
- `tests/test_cutcell_eval.py` — evaluator cache, ψ=1 ≡ uniform
  identity, ψ-parameterized checks, ψ sweeps (E4_1 variants `[slow]`).
- `tests/test_reflection.py` — mode split, R(s) properties (row-scaling
  invariance, singularity equivalence, E2 analytic anchor, sonic
  routing), mirror checks, ψ-swept reflection, round-trip (Gaussian
  acceptance, E2_1 pairs, Δj minimum, p=2 ordering tripwire), report
  wiring (`TestCutcellStabilityReport`).
- `tests/test_interval_operator.py` — assembler identities, row-sum
  consistency, minimum-N, scan cross-checks and outflow law.
- `tests/test_optimizer.py::TestExtractField` — dotted-path extraction
  of `layer_cutcell` fields.

Focused run (from `scripts/stencil_gen/`):

```bash
SYMPY_CACHE_SIZE=50000 uv run pytest tests/ -x -q -k "cutcell or reflection or interval"
```
