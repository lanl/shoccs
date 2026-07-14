"""Tests for RK4 stability-region boundary containment (plan 48 item 4.1).

Plan-prescribed tests:

(a) L = -I (scalar decay): ``sigma_min(z I - dt L) = |z + dt|`` exactly,
    so the margin must equal ``min |z + dt|`` over the cached boundary
    polyline (hand-derivable anchor);
(b) a normal advection circulant at CFL 0.8: eigenvalues inside,
    margin > 0 — asserted on the mean-zero deflation, because the raw
    periodic circulant's constant mode sits exactly ON dS at zeta = 0
    and honestly zeroes the margin (S is closed; also asserted); at
    CFL 3.0 ``eigs_inside is False``;
(c) scale test: a non-normal perturbation that inflates the resolvent
    norm shrinks the margin (qualitative: assert margin decreases;
    observed ~halving when the perturbation doubles).

Plus: boundary-polyline verification (level set |R| = 1, closure,
winding = 4 around every zero of R, conjugate symmetry, known anchors
x* ~ -2.7853 and +-2 sqrt(2) i, region coverage), module-level caching /
immutability, Lipschitz-certified margin bookkeeping, and the cut-cell
psi-sweep wiring on E2_1 (test names contain ``cutcell``).  E4_1 tests
are @pytest.mark.slow (symbolic derivation > 170 s, plan 48 scheme-scope
constraint).  **No time integration anywhere** — resolvent quantities
only (plan 48 hard constraint).
"""

import numpy as np
import pytest

from stencil_gen.cutcell_eval import make_cutcell_evaluator
from stencil_gen.fully_discrete import (
    ContainmentResult,
    Rk4ContainmentSweep,
    rk4_amplification,
    rk4_containment,
    rk4_containment_over_psi,
    rk4_stability_boundary,
)
from stencil_gen.interval_operator import interval_stability_scan
from stencil_gen.temo import E2_1, E4_1

# RK4 real stability limit: the negative real root of R(x) = 1, i.e. of
# the cubic x^3/24 + x^2/6 + x/2 + 1 = 0 (x* ~ -2.7853).
_CUBIC = [1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0]
X_STAR = float(
    next(r.real for r in np.roots(_CUBIC) if abs(r.imag) < 1e-12)
)


def _upwind_circulant(n: int) -> np.ndarray:
    """First-order upwind advection semi-discrete operator L = -D at unit
    spacing and speed: (L u)_j = -(u_j - u_{j-1}), periodic.  A circulant,
    hence normal; eigenvalues -(1 - e^{-i theta_k}) on the circle of
    radius 1 centered at -1 (the constant mode k = 0 maps to exactly 0)."""
    D = np.eye(n) - np.roll(np.eye(n), 1, axis=1).T
    return -D


def _deflate_constant_mode(L: np.ndarray, seed: int = 0) -> np.ndarray:
    """Restrict the circulant to the mean-zero invariant subspace (the
    orthogonal complement of the constant eigenvector), removing the
    zeta = 0 boundary mode; the restriction of a normal operator to an
    invariant subspace is normal."""
    n = L.shape[0]
    v = np.ones(n) / np.sqrt(n)
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(np.column_stack([v, rng.standard_normal((n, n - 1))]))
    Qc = Q[:, 1:]  # first QR column spans v exactly
    return Qc.T @ L @ Qc


@pytest.fixture(scope="module")
def boundary_1024():
    return rk4_stability_boundary(1024)


@pytest.fixture(scope="module")
def e2_evaluator():
    """Module-cached E2_1 evaluator (lru_cache makes this near-free)."""
    return make_cutcell_evaluator(E2_1)


@pytest.fixture(scope="module")
def e2_alpha_zero(e2_evaluator):
    return {s: 0.0 for s in e2_evaluator.alpha_symbols}


@pytest.fixture(scope="module")
def e2_alpha_nonzero(e2_evaluator):
    """The interval-test alpha point that breaks the alpha = 0 row
    degeneracy; eigenvalue-UNSTABLE on the cut interval (max Re ~ +1.78)
    — used here as the flagged-unstable sweep case."""
    return dict(zip(e2_evaluator.alpha_symbols, [0.3, -0.2, 0.15, 0.05]))


@pytest.fixture(scope="module")
def e2_scan_stable(e2_alpha_zero):
    """Tiny precomputed item-3.2 scan (cut-left/uniform-right, the stable
    half-line-like configuration) whose worst triple is (0.3, 1.0, 8)."""
    return interval_stability_scan(
        E2_1, e2_alpha_zero, psi_pairs=[(0.3, 1.0)], N_values=(8,)
    )


class TestRk4Boundary:
    """The cached dS polyline: level set, closure, winding, symmetry,
    known anchors, coverage, caching."""

    def test_polyline_on_level_set_and_closed(self, boundary_1024):
        b = boundary_1024
        assert b.shape == (1024,)
        assert b[0] == 0.0 + 0.0j  # march starts exactly at the origin
        # Every sample sits on |R(zeta)| = 1.
        np.testing.assert_allclose(
            np.abs(rk4_amplification(b)), 1.0, rtol=0, atol=1e-12
        )
        # Closed loop: the wrap-around gap is a single-step gap, not a
        # quarter-boundary chasm (the theta in [0, 2 pi) recipe traces
        # only 1/4 of dS — arg R winds 4x; see module docstring).
        gaps = np.abs(np.diff(b))
        wrap = abs(b[0] - b[-1])
        assert wrap < 3.0 * np.median(gaps)
        assert max(gaps.max(), wrap) < 0.03  # perimeter ~14.84 / 1024

    def test_winding_is_one_around_every_zero_of_R(self, boundary_1024):
        """The traced loop encloses all four zeros of R exactly once —
        the argument-principle bookkeeping behind the 8 pi march (a
        quarter trace would give winding 1/4-ish and a false margin)."""
        zeros = np.roots([1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0])
        for z0 in zeros:
            v = boundary_1024 - z0
            ang = np.angle(np.roll(v, -1) / v)
            w = ang.sum() / (2.0 * np.pi)
            assert w == pytest.approx(1.0, abs=1e-6)

    def test_conjugate_symmetry(self, boundary_1024):
        """R has real coefficients, so dS is symmetric about the real
        axis; the march parameterization pairs sample j with n - j."""
        b = boundary_1024
        n = b.size
        idx = (n - np.arange(n)) % n
        np.testing.assert_allclose(b[idx], np.conj(b), rtol=0, atol=1e-9)

    def test_known_anchors(self, boundary_1024):
        b = boundary_1024
        # Half-way sample (theta = 4 pi, target e^{i theta} = 1) is the
        # negative-real crossing: the RK4 real stability limit x*.
        assert b[512].imag == pytest.approx(0.0, abs=1e-12)
        assert b[512].real == pytest.approx(X_STAR, abs=1e-9)
        assert b.real.min() == pytest.approx(X_STAR, abs=1e-9)
        # +-2 sqrt(2) i lies on the level set (|R(2 sqrt(2) i)| = 1
        # exactly: R = -1/3 - (2 sqrt(2)/3) i there) and the boundary
        # reaches above it (the RK4 lobe tops out near 2.94 i).
        y = 2.0 * np.sqrt(2.0)
        assert abs(rk4_amplification(1j * y)) == pytest.approx(1.0, abs=1e-12)
        assert b.imag.max() > y
        assert b.imag.max() < 3.1
        # S pokes into the right half-plane near the imaginary axis (the
        # well-known RK4 sliver): the boundary has Re zeta > 0 samples.
        assert 0.2 < b.real.max() < 0.3

    def test_region_coverage(self):
        """Every grid point with |R| <= 1 (away from the boundary) lies
        inside the traced polygon, and every |R| > 1 point outside it:
        the polyline is the boundary of the WHOLE stability set, so the
        margin is never taken over a partial curve."""
        poly = rk4_stability_boundary(256)
        spacing = max(np.abs(np.diff(poly)).max(), abs(poly[0] - poly[-1]))
        xs = np.linspace(-3.2, 0.6, 39)
        ys = np.linspace(-3.1, 3.1, 63)
        X, Y = np.meshgrid(xs, ys)
        Z = (X + 1j * Y).ravel()
        dmin = np.min(np.abs(Z[:, None] - poly[None, :]), axis=1)
        far = dmin > 2.0 * spacing  # skip the ambiguous boundary collar
        Zf = Z[far]
        v = poly[None, :] - Zf[:, None]
        ang = np.angle(np.roll(v, -1, axis=1) / v)
        w = np.abs(ang.sum(axis=1)) / (2.0 * np.pi)
        inside_poly = w > 0.5
        in_S = np.abs(rk4_amplification(Zf)) <= 1.0
        np.testing.assert_array_equal(inside_poly, in_S)

    def test_cached_and_readonly(self, boundary_1024):
        assert rk4_stability_boundary(1024) is boundary_1024  # lru_cache
        assert not boundary_1024.flags.writeable
        with pytest.raises((ValueError, RuntimeError)):
            boundary_1024[0] = 1.0

    def test_minimum_n_boundary(self):
        assert rk4_stability_boundary(128).shape == (128,)
        with pytest.raises(ValueError, match="n_boundary"):
            rk4_stability_boundary(127)


class TestRk4Containment:
    """Plan tests (a)-(c) plus bookkeeping and validation."""

    def test_scalar_decay_margin_matches_polyline(self, boundary_1024):
        """Plan test (a): for L = -I, sigma_min(z I - dt L) = |z + dt|,
        so the margin is exactly min |z + dt| over the polyline."""
        dt = 0.7
        res = rk4_containment(-np.eye(3), dt)
        expected = float(np.min(np.abs(boundary_1024 + dt)))
        assert res.margin == pytest.approx(expected, rel=1e-12)
        # The nearest boundary point to -0.7 is the origin sample.
        assert res.argmin_z == 0.0 + 0.0j
        assert res.margin == pytest.approx(dt, rel=1e-12)
        assert res.eigs_inside is True
        np.testing.assert_allclose(res.eigenvalues, -dt, atol=1e-14)
        # Lipschitz-certified value: margin - max_arc_spacing / 2.
        spacing = max(
            np.abs(np.diff(boundary_1024)).max(),
            abs(boundary_1024[0] - boundary_1024[-1]),
        )
        assert res.max_arc_spacing == pytest.approx(spacing, rel=1e-12)
        assert res.margin_certified == pytest.approx(
            res.margin - spacing / 2.0, rel=1e-12
        )
        assert res.margin_certified < res.margin
        assert res.n_boundary == 1024

    def test_scalar_decay_real_axis_limit(self):
        """dt L crossing x* ~ -2.7853 flips eigs_inside (S is closed and
        bounded on the negative real axis by the real root of R = 1)."""
        assert rk4_containment(-np.eye(2), 2.5).eigs_inside is True
        assert rk4_containment(-np.eye(2), 3.0).eigs_inside is False

    def test_certified_margin_tightens_with_n_boundary(self):
        """At 256 points the spacing correction (~0.05) is material —
        the plan's discretization-honesty note — and shrinks at 1024."""
        res256 = rk4_containment(-np.eye(2), 0.7, n_boundary=256)
        res1024 = rk4_containment(-np.eye(2), 0.7, n_boundary=1024)
        assert res256.max_arc_spacing / 2.0 == pytest.approx(0.05, abs=0.01)
        assert res1024.margin_certified > res256.margin_certified
        assert res1024.margin == pytest.approx(res256.margin, rel=1e-6)

    def test_advection_circulant_cfl(self):
        """Plan test (b): normal advection circulant at CFL 0.8 —
        eigenvalues inside and margin > 0 (mean-zero deflation); at
        CFL 3.0 eigs_inside is False.  The RAW circulant's constant mode
        sits exactly ON dS at zeta = 0 (|R(0)| = 1): still counted
        inside (S is closed) but the margin is honestly zero — periodic
        advection is only neutrally stable under RK4."""
        L = _upwind_circulant(9)

        res_raw = rk4_containment(L, 0.8)
        assert res_raw.eigs_inside is True
        assert res_raw.margin == pytest.approx(0.0, abs=1e-12)

        Lr = _deflate_constant_mode(L)
        res = rk4_containment(Lr, 0.8)
        assert res.eigs_inside is True
        assert res.margin > 0.1  # measured 0.187
        assert res.margin_certified > 0.1  # 0.175 at n_boundary=1024

        assert rk4_containment(L, 3.0).eigs_inside is False
        assert rk4_containment(Lr, 3.0).eigs_inside is False

    def test_nonnormal_perturbation_shrinks_margin(self):
        """Plan test (c): the spectrum stays fixed at -1 (inside S) while
        a nilpotent perturbation inflates the resolvent norm along dS;
        the containment margin must decrease (observed ~halving when the
        perturbation doubles: 0.236 -> 0.123 for c = 4 -> 8)."""
        margins = {}
        for c in (0.0, 4.0, 8.0):
            Lc = np.array([[-1.0, c], [0.0, -1.0]])
            res = rk4_containment(Lc, 1.0)
            assert res.eigs_inside is True  # spectrum unchanged
            margins[c] = res.margin
        assert margins[8.0] < margins[4.0] < margins[0.0]
        assert margins[8.0] / margins[4.0] < 0.6  # ~halved

    def test_validation(self):
        with pytest.raises(ValueError, match="square"):
            rk4_containment(np.zeros((2, 3)), 0.5)
        with pytest.raises(ValueError, match="dt"):
            rk4_containment(-np.eye(2), 0.0)
        with pytest.raises(ValueError, match="dt"):
            rk4_containment(-np.eye(2), -0.1)
        with pytest.raises(ValueError, match="n_boundary"):
            rk4_containment(-np.eye(2), 0.5, n_boundary=64)


class TestRk4ContainmentSweepCutcell:
    """psi-swept containment for cut-cell interval operators (E2_1).

    Semi-discrete stability context (from the item-3.1/3.2 tests): under
    ``u_t = -D u`` the cut-left + uniform-right interval is the stable
    half-line-like family, while a cut Dirichlet closure at the RIGHT
    (outflow) end carries the eigenvalue (1 - psi_R)/(1 + psi_R) > 0 —
    which is exactly what the worst-scan-triple evaluation must flag.
    """

    def test_cutcell_sweep_structure_and_honest_zero_margin(
            self, e2_alpha_zero, e2_scan_stable):
        """Wiring + the alpha = 0 honesty case: the E2_1 alpha = 0
        closure has coinciding Dirichlet rows 0 and 1, so the interval
        operator carries an EXACT zero eigenvalue — on dS at zeta = 0 —
        and every containment margin is honestly ~0 while eigs_inside
        stays True (S is closed).  The zero is DEFECTIVE (Jordan block),
        so its computed eigenvalues split by ~sqrt(eps) ~ 1e-8; the
        explicit eig_tol keeps the closed-region test honest about that
        (module docstring caveat) — the load-bearing number here is the
        zero margin, not the boundary-mode classification."""
        sweep = rk4_containment_over_psi(
            E2_1, e2_alpha_zero, 0.8, 0.1, [0.3, 0.5],
            N=12, n_boundary=256, scan=e2_scan_stable, eig_tol=1e-6,
        )
        assert isinstance(sweep, Rk4ContainmentSweep)
        assert set(sweep.per_psi) == {0.3, 0.5}
        assert set(sweep.scan_cases) == {(0.3, 1.0, 8)}
        assert sweep.N == 12 and sweep.psi_R == 1.0
        assert sweep.cfl == 0.8 and sweep.h == 0.1
        assert sweep.dt == pytest.approx(0.08, rel=1e-15)
        assert sweep.infeasible_psi == []

        for res in [*sweep.per_psi.values(), *sweep.scan_cases.values()]:
            assert isinstance(res, ContainmentResult)
            assert res.eigs_inside is True
            assert res.margin == pytest.approx(0.0, abs=1e-12)
        assert sweep.any_eigs_outside is False
        assert sweep.worst_margin == pytest.approx(0.0, abs=1e-12)
        spacing = max(
            np.abs(np.diff(rk4_stability_boundary(256))).max(),
            abs(rk4_stability_boundary(256)[0]
                - rk4_stability_boundary(256)[-1]),
        )
        assert sweep.worst_margin_certified == pytest.approx(
            -spacing / 2.0, abs=1e-12
        )
        assert sweep.worst_case in {0.3, 0.5, (0.3, 1.0, 8)}

    def test_cutcell_sweep_h_cancels(self, e2_alpha_nonzero,
                                     e2_scan_stable):
        """dt L = -cfl D up to rounding (unit-spacing assembler): the
        margins are h-independent even though dt itself scales with h.
        Uses the non-degenerate alpha point so the compared margins are
        O(1e-3), far above rounding."""
        kw = dict(N=12, n_boundary=256, scan=e2_scan_stable)
        a = rk4_containment_over_psi(
            E2_1, e2_alpha_nonzero, 0.8, 0.1, [0.5], **kw
        )
        b = rk4_containment_over_psi(
            E2_1, e2_alpha_nonzero, 0.8, 0.025, [0.5], **kw
        )
        assert a.dt == pytest.approx(0.08) and b.dt == pytest.approx(0.02)
        assert a.per_psi[0.5].margin == pytest.approx(
            b.per_psi[0.5].margin, rel=1e-9
        )
        assert a.per_psi[0.5].eigs_inside == b.per_psi[0.5].eigs_inside

    def test_cutcell_sweep_worst_case_aggregation_unstable_alpha(
            self, e2_alpha_nonzero):
        """The eigenvalue-unstable alpha point: every case has eigenvalues
        far outside S (max Re dt lambda ~ 1.4), any_eigs_outside is True,
        and the headline worst-case bookkeeping picks the smallest
        certified margin across BOTH families (here psi = 0.3, measured
        margins 2.08e-3 < 4.48e-3 < 7.13e-3)."""
        scan = interval_stability_scan(
            E2_1, e2_alpha_nonzero, psi_pairs=[(0.3, 1.0)], N_values=(8,)
        )
        sweep = rk4_containment_over_psi(
            E2_1, e2_alpha_nonzero, 0.8, 0.1, [0.3, 0.5],
            N=12, n_boundary=256, scan=scan,
        )
        assert sweep.any_eigs_outside is True
        for res in [*sweep.per_psi.values(), *sweep.scan_cases.values()]:
            assert res.eigs_inside is False
        assert sweep.per_psi[0.3].margin < sweep.per_psi[0.5].margin
        assert (sweep.per_psi[0.3].margin
                < sweep.scan_cases[(0.3, 1.0, 8)].margin)
        assert sweep.worst_case == 0.3
        assert sweep.worst_margin == sweep.per_psi[0.3].margin
        assert (sweep.worst_margin_certified
                == sweep.per_psi[0.3].margin_certified)

    def test_cutcell_sweep_scan_triple_flags_outflow_resonance(
            self, e2_alpha_zero):
        """A fixed-N inflow-cut sweep alone would miss the outflow-cut
        instability (eigenvalue (1 - psi_R)/(1 + psi_R) = 1/3 at
        psi_R = 0.5, item-3.1 test finding); the mandatory worst-scan-
        triple evaluation catches it: dt lambda ~ 0.8/3 is far outside
        S while the sweep family stays inside."""
        scan = interval_stability_scan(
            E2_1, e2_alpha_zero, psi_pairs=[(0.5, 0.5)], N_values=(8,)
        )
        assert scan.worst_abscissa == pytest.approx(1.0 / 3.0, rel=1e-10)
        sweep = rk4_containment_over_psi(
            E2_1, e2_alpha_zero, 0.8, 0.1, [0.5],
            N=12, n_boundary=256, scan=scan, eig_tol=1e-6,
        )
        assert sweep.per_psi[0.5].eigs_inside is True
        assert (0.5, 0.5, 8) in sweep.scan_cases
        assert sweep.scan_cases[(0.5, 0.5, 8)].eigs_inside is False
        assert sweep.any_eigs_outside is True

    def test_cutcell_sweep_internal_scan_via_scan_kwargs(
            self, e2_alpha_zero):
        """Without a precomputed scan the sweep runs interval_stability_
        scan itself with the given kwargs and evaluates its worst
        triple(s)."""
        sweep = rk4_containment_over_psi(
            E2_1, e2_alpha_zero, 0.8, 0.1, [0.5],
            N=12, n_boundary=256,
            scan_kwargs=dict(psi_pairs=[(0.3, 1.0)], N_values=(8,)),
        )
        assert set(sweep.scan_cases) == {(0.3, 1.0, 8)}

    def test_cutcell_sweep_infeasible_partition(self, e2_alpha_zero):
        """Infeasible psi points (item-1.2 guard) are partitioned out —
        never folded into margins or verdicts.  E2_1's structural
        denominator (psi + 1) never vanishes on (0, 1], so an inflated
        tolerance stands in for a vanishing one (same device as the
        item-3.2 tests)."""
        scan = interval_stability_scan(
            E2_1, e2_alpha_zero, psi_pairs=[(0.5, 0.5)], N_values=(8,),
            feasibility_tol=10.0,
        )
        sweep = rk4_containment_over_psi(
            E2_1, e2_alpha_zero, 0.8, 0.1, [0.3, 0.5],
            N=12, n_boundary=256, scan=scan, feasibility_tol=10.0,
        )
        assert sweep.infeasible_psi == [0.3, 0.5]
        assert sweep.per_psi == {}
        assert sweep.scan_cases == {}
        assert sweep.worst_margin == float("inf")
        assert sweep.worst_margin_certified == float("inf")
        assert sweep.worst_case is None
        assert sweep.any_eigs_outside is False

    def test_cutcell_sweep_validation(self, e2_alpha_zero, e2_scan_stable):
        kw = dict(N=12, n_boundary=256, scan=e2_scan_stable)
        with pytest.raises(ValueError, match="cfl"):
            rk4_containment_over_psi(
                E2_1, e2_alpha_zero, 0.0, 0.1, [0.5], **kw)
        with pytest.raises(ValueError, match="h must"):
            rk4_containment_over_psi(
                E2_1, e2_alpha_zero, 0.8, -0.1, [0.5], **kw)
        with pytest.raises(ValueError, match="psi_R"):
            rk4_containment_over_psi(
                E2_1, e2_alpha_zero, 0.8, 0.1, [0.5], psi_R=1.5, **kw)
        with pytest.raises(ValueError, match="psi_grid"):
            rk4_containment_over_psi(
                E2_1, e2_alpha_zero, 0.8, 0.1, [], **kw)
        with pytest.raises(ValueError, match="psi must"):
            rk4_containment_over_psi(
                E2_1, e2_alpha_zero, 0.8, 0.1, [0.0], **kw)
        with pytest.raises(TypeError, match="not both"):
            rk4_containment_over_psi(
                E2_1, e2_alpha_zero, 0.8, 0.1, [0.5],
                N=12, scan=e2_scan_stable, scan_kwargs={},
            )


@pytest.mark.slow
class TestE4_1SweepSlow:
    """E4_1 smoke test — symbolic derivation > 170 s, slow-marked
    (plan 48 scheme-scope constraint)."""

    def test_cutcell_sweep_e4_1_structure(self):
        ev = make_cutcell_evaluator(E4_1)
        alpha = {s: 0.0 for s in ev.alpha_symbols}
        sweep = rk4_containment_over_psi(
            E4_1, alpha, 0.8, 1.0, [0.5],
            N=13, n_boundary=256,
            scan_kwargs=dict(psi_pairs=[(0.5, 0.5)], N_values=(8,)),
        )
        assert set(sweep.per_psi) == {0.5}
        assert (0.5, 0.5, 8) in sweep.scan_cases
        for res in [*sweep.per_psi.values(), *sweep.scan_cases.values()]:
            assert np.isfinite(res.margin)
            assert np.isfinite(res.margin_certified)
            assert res.eigenvalues.shape[0] in (13, 8)
