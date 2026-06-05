# Brady-Livescu §4.2 Reflecting-Hyperbolic Eigenvalue Layer (L3r / BL42)

## Problem statement

From Brady & Livescu 2019 §4.2 (pp. 91–92): a 1D coupled hyperbolic
system with energy-conserving reflecting boundary conditions.

```
u_t = v_x
v_t = u_x          on [0, 1],  t in [0, 500]

u(0, t) = 0         (Dirichlet, reflecting)
v(1, t) = 0         (Dirichlet, reflecting)

u(x, 0) = -(3*pi/2) * sin(3*pi*x/2)
v(x, 0) = 0
```

Exact solution (standing-wave superposition from d'Alembert decomposition):

```
u(x, t) = -(3*pi/2) * sin(3*pi*x/2) * cos(3*pi*t/2)
v(x, t) = -(3*pi/2) * cos(3*pi*x/2) * sin(3*pi*t/2)
```

The initial condition excites a single eigenmode (k=2, ω = 3π/2).

## Why it's the cleanest boundary-closure stability test

Unlike the §4.3 varying-coefficient advection problem (L7), where
`div(c) = 1/ψ > 0` introduces physical growth that must be calibrated
out (`L7_TOL = 5e-3`), the §4.2 system has three properties that make
it an ideal discriminator:

1. **`div(c) = 0`** — the coefficient matrix is constant and symmetric.
2. **Energy-conserving BCs** — reflecting Dirichlet conditions satisfy
   `⟨Lu, u⟩ = 0` in the continuous L²; no dissipation.
3. **Purely imaginary continuous spectrum** — eigenvalues are
   `λ_k = ±i(2k−1)π/2` for positive integer k.

Any discrete `Re(λ) > tol` is therefore an unambiguous signature of
boundary-closure instability, with no calibration needed. The tolerance
`BL42_TOL = 1e-10` is tight and justified.

## Block operator construction

Semi-discrete form with a 1D differentiation matrix `D` (shape N×N):

```
dq/dt = L q,   q = [u; v],   L = [[0, D/h], [D/h, 0]]
```

where `h = L_DOMAIN / (N - 1) = 1 / (N - 1)`.

After removing the Dirichlet DOFs — row/col 0 (`u` at `x = 0`) and
row/col `2N − 1` (`v` at `x = 1`) — the reduced operator `L_red` has
shape `(2N − 2) × (2N − 2)`.

## API reference

### `build_bl42_operator(D) -> scipy.sparse.csr_matrix`

In `stencil_gen/brady2d_stability.py`. Builds the reduced `(2N-2) × (2N-2)`
block operator from a 1D differentiation matrix `D` of shape `(N, N)`.

- Scales `D` by `1/h` where `h = 1 / (N - 1)`.
- Constructs `L = [[0, D/h], [D/h, 0]]` via `scipy.sparse.bmat`.
- Removes DOFs at indices 0 and 2N−1 (Dirichlet rows/columns).

### `layer_bl42_reflecting_hyperbolic(scheme, kernel, params, n_values=(21, 41, 81)) -> dict`

Layer function that evaluates BL42 eigenvalue stability at multiple grid
sizes.

Returns:

| Key | Type | Description |
|-----|------|-------------|
| `spectral_abscissa_by_n` | `dict[int, float]` | `{N: max Re(λ)}` for each grid size |
| `max_spectral_abscissa` | `float` | Maximum over all grid sizes |
| `purely_imaginary` | `bool` | `True` iff `max_spectral_abscissa < BL42_TOL` |

### `BL42_TOL = 1e-10`

Failure threshold. The continuous spectrum is exactly imaginary, so any
positive real part above machine noise indicates instability.

### Reference problem module

`stencil_gen/benchmarks/brady_livescu_4_2.py` provides:

| Function | Description |
|----------|-------------|
| `initial_u(x)` | IC: `-(3π/2) sin(3πx/2)` |
| `initial_v(x)` | IC: zeros |
| `exact_solution(x, t)` | Standing-wave closed form `(u, v)` |
| `continuous_eigenvalues(k_max=20)` | `±i(2k−1)π/2` for `k = 1..k_max` |
| `L_DOMAIN = 1.0` | Domain length constant |

## Cascade position

L3r runs during the L3 tier — after L3 (1D advection eigenvalue) and
before L4 (2D local group velocity). It is a parallel 1D eigenvalue
check on a different model problem.

```
L1 (GV error) → L2 (Kreiss) → L3 (advection eig) → L3r (BL42 eig) → L4 → L5 → L6 → L7 → L8
```

On failure:
- `failed_layer = 3` (grouped with L3 for gate purposes).
- `failed_reason` contains `"BL42"` to disambiguate from L3 failures.
- A `gate_layer=3` feasibility cliff requires both L3 **and** L3r to pass.

L3r runs whenever `max_layer >= 3`. It is strictly cheaper than L7
(≤ 2N ≈ 160 dimensions at N=80, vs. (N−1)² ≈ 6400 for L7).

### `StabilityReport` field

```python
layer_bl42: dict | None = None
```

The `__str__` output line:
```
L3r BL42 reflecting: PASS  max_re=6.0025e-14  per_n=[21:1.23e-14, 41:6.00e-14, 81:5.50e-14]
```

### Failure threshold table

| Layer | Metric | Threshold | Value | Rationale |
|-------|--------|-----------|-------|-----------|
| L3r | `max_spectral_abscissa` | `BL42_TOL` | 1e-10 | Continuous spectrum is exactly imaginary |

## Optimizer integration

BL42 fields are valid optimizer objectives via the standard dotted-path
convention:

```bash
cd scripts/stencil_gen
uv run python -m sweeps optimize \
    --scheme E4 --kernel tension \
    --objective layer_bl42.max_spectral_abscissa \
    --bounds 0.5 20 --method Nelder-Mead --max-evals 40
```

The `_FIELD_LAYER_ALIAS` mapping `"layer_bl42": 3` ensures that
`make_objective` infers `max_layer=3` for BL42 objectives.

When using `gate_layer`, note that tension kernels fail L3r at
`gate_layer=3`. Use `--gate-layer 2` to gate only on L1-L2 while still
computing the BL42 spectral abscissa as the optimization target.

## Calibration results

From `sweeps/known_values.json` under `"brady2d_calibration"`:

| Family | Verdict | `max_spectral_abscissa` | `purely_imaginary` |
|--------|---------|------------------------|--------------------|
| E4\_classical | **pass** | 6.00e-14 | true |
| E2\_phs\_k2 | **pass** | 4.89e-15 | true |
| E4\_phs\_k2 | fail (L3r) | 6.30e-01 | false |
| E4\_tension\_3 | fail (L3r) | 9.54e-01 | false |
| E4\_gaussian\_09 | fail (L3r) | 7.00e-01 | false |
| E4\_multiquadric\_1 | fail (L3r) | 6.48e-01 | false |
| E2\_tension\_6 | fail (L1) | — | — |
| E2\_gaussian\_2 | fail (L1) | — | — |
| E2\_multiquadric\_1 | fail (L1) | — | — |

Only 2 of 9 families pass BL42, compared to 6 of 9 that pass the L3
advection eigenvalue check. This confirms BL42's value as a stricter
discriminator: schemes that are stable for constant-coefficient
advection can still fail on the reflecting-BC hyperbolic system.

## References

- Brady, P.T. and Livescu, D. (2019). "High-order, stable, and
  conservative boundary schemes for central and compact finite
  differences." *Computers & Fluids*, 183, pp. 84–101. §4.2, pp. 91–92.
