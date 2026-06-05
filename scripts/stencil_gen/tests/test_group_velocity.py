"""Tests for group velocity analysis module."""

import numpy as np
import pytest

from stencil_gen.group_velocity import (
    AnisotropyResult,
    GKSModeInfo,
    GroupVelocity2DResult,
    GroupVelocityProfile,
    PsiSweepResult,
    anisotropy_profile,
    boundary_group_velocity,
    boundary_group_velocity_2d,
    boundary_group_velocity_classical,
    cut_cell_group_velocity,
    gks_group_velocity_check,
    group_velocity_numerical,
    group_velocity_2d,
    group_velocity_error,
    group_velocity_exact,
    group_velocity_exact_nonuniform,
    interior_group_velocity,
    local_group_velocity,
    anisotropy_over_coefficient_field,
    local_group_velocity_2d_varying,
    max_local_gv_error_2d,
    modified_wavenumber,
    modified_wavenumber_nonuniform,
    phase_velocity,
    psi_sweep_group_velocity,
    ray_trace_group_velocity,
)


class TestCoreGroupVelocity:
    """Core group velocity computation tests (34.1b)."""

    N_XI = 500

    def test_exact_scheme_unity_group_velocity(self):
        """For exact differentiation (central difference of delta), C(xi) = 1."""
        # For D1 approximating d/dx, kappa*(xi) has Im(kappa*) approximating xi.
        # Group velocity: C(xi) = d(Im(kappa*))/d(xi).
        # For exact differentiation, Im(kappa*) = xi, so C = 1.
        #
        # No finite stencil gives kappa* = i*xi exactly, so we test the
        # "spectral" limit: use a high-order scheme (E8) and verify C ≈ 1
        # at low wavenumbers where the truncation error is negligible.
        from stencil_gen.interior import derive_interior, full_gamma_array

        p = 4  # E8 scheme
        coeffs = derive_interior(0, p, 1)
        w = [float(c) for c in full_gamma_array(coeffs)]
        nodes = list(range(-p, p + 1))
        xi = np.linspace(0.01, 0.5, self.N_XI)  # well-resolved wavenumbers

        C = group_velocity_exact(w, 0, nodes, xi)
        # At low xi, C should be very close to 1 for high-order scheme
        assert np.allclose(C, 1.0, atol=1e-4), (
            f"E8 group velocity should be ~1 at low xi, got max error "
            f"{np.max(np.abs(C - 1.0)):.2e}"
        )

    def test_numerical_vs_analytical_gradient(self):
        """group_velocity_numerical() and group_velocity_exact() agree within numerical tolerance."""
        # Use E2 interior stencil: weights [-1/2, 0, 1/2]
        weights = [-0.5, 0.0, 0.5]
        nodes = [-1, 0, 1]
        xi = np.linspace(0.01, np.pi - 0.01, self.N_XI)

        kstar = modified_wavenumber(weights, 0, nodes, xi)
        C_numerical = group_velocity_numerical(kstar, xi)
        C_analytical = group_velocity_exact(weights, 0, nodes, xi)

        # Numerical differentiation (2nd-order central diff) agrees to ~O(h^2)
        assert np.allclose(C_numerical, C_analytical, atol=1e-4), (
            f"Numerical vs analytical group velocity max diff: "
            f"{np.max(np.abs(C_numerical - C_analytical)):.2e}"
        )

    def test_phase_velocity_low_xi_limit(self):
        """Phase velocity c(xi) -> 1 as xi -> 0 for any consistent scheme."""
        from stencil_gen.interior import derive_interior, full_gamma_array

        for p in [1, 2, 3]:
            coeffs = derive_interior(0, p, 1)
            w = [float(c) for c in full_gamma_array(coeffs)]
            nodes = list(range(-p, p + 1))
            xi = np.linspace(0.01, 0.1, 50)

            kstar = modified_wavenumber(w, 0, nodes, xi)
            c = phase_velocity(kstar, xi)

            # At small xi, phase velocity should be close to 1
            assert abs(c[0] - 1.0) < 1e-3, (
                f"p={p}: phase velocity at xi={xi[0]:.3f} is {c[0]:.6f}, "
                f"expected ~1.0"
            )

    def test_second_order_known_values(self):
        """For E2 interior stencil [-1/2, 0, 1/2], C(xi) = cos(xi)."""
        weights = [-0.5, 0.0, 0.5]
        nodes = [-1, 0, 1]
        xi = np.linspace(0, np.pi, self.N_XI)

        # Analytical group velocity
        C = group_velocity_exact(weights, 0, nodes, xi)
        expected = np.cos(xi)

        assert np.allclose(C, expected, atol=1e-14), (
            f"E2 group velocity should be cos(xi), max error: "
            f"{np.max(np.abs(C - expected)):.2e}"
        )

        # Also verify via modified wavenumber
        kstar = modified_wavenumber(weights, 0, nodes, xi)
        # kappa* = i*sin(xi), so Im(kappa*) = sin(xi)
        assert np.allclose(np.imag(kstar), np.sin(xi), atol=1e-14)
        assert np.allclose(np.real(kstar), 0.0, atol=1e-14)

    def test_interior_group_velocity_e2(self):
        """Smoke test for interior_group_velocity(): E2 (p=1) returns correct profile."""
        xi = np.linspace(0, np.pi, self.N_XI)
        profile = interior_group_velocity(p=1, nu=1, xi_array=xi)

        # Correct type and order
        assert isinstance(profile, GroupVelocityProfile)
        assert profile.order == 2

        # E2 group velocity is cos(xi)
        assert np.allclose(profile.group_velocity, np.cos(xi), atol=1e-14)

        # Cutoff where C first goes to zero: cos(xi)=0 at xi=pi/2
        assert abs(profile.cutoff_xi - np.pi / 2) < 2 * np.pi / self.N_XI


class TestInteriorGroupVelocity:
    """Interior scheme group velocity analysis (34.2b)."""

    N_XI = 2000

    def test_error_amplification_factor(self):
        """Group velocity error is (2p+1) times phase velocity error at leading order.

        For a 2p-th order scheme, Im(kappa*) = xi - a*xi^(2p+1) + ..., so:
          phase velocity error:  c - 1 = -a*xi^(2p) + ...
          group velocity error:  C - 1 = -(2p+1)*a*xi^(2p) + ...
        The leading-order ratio is (2p+1).

        Note: the original plan stated (2p-1), but the correct factor from
        the Taylor expansion is (2p+1).  Verified numerically below.
        """
        xi = np.linspace(0.01, 0.15, 500)
        for p in [1, 2, 3, 4]:
            profile = interior_group_velocity(p=p, nu=1, xi_array=xi)
            gv_err = profile.group_velocity - 1.0
            pv_err = profile.phase_velocity - 1.0
            # Use a point near the middle where errors are small but nonzero
            idx = len(xi) // 2
            ratio = gv_err[idx] / pv_err[idx]
            expected = 2 * p + 1
            assert abs(ratio - expected) < 0.1, (
                f"E{2*p}: gv_err/pv_err ratio = {ratio:.2f}, expected {expected}"
            )

    def test_cutoff_wavenumber(self):
        """Cutoff xi (where C = 0) increases with order.

        Higher-order schemes resolve more wavenumbers before group velocity
        reversal, so cutoff_xi moves to higher values.
        """
        xi = np.linspace(0, np.pi, self.N_XI)
        cutoffs = []
        for p in [1, 2, 3, 4]:
            profile = interior_group_velocity(p=p, nu=1, xi_array=xi)
            cutoffs.append(profile.cutoff_xi)

        # Cutoffs should be strictly increasing
        for i in range(len(cutoffs) - 1):
            assert cutoffs[i] < cutoffs[i + 1], (
                f"E{2*(i+1)} cutoff {cutoffs[i]:.4f} >= "
                f"E{2*(i+2)} cutoff {cutoffs[i+1]:.4f}"
            )

        # E2 cutoff should be pi/2
        assert abs(cutoffs[0] - np.pi / 2) < 2 * np.pi / self.N_XI

    def test_group_velocity_sign_reversal(self):
        """C(xi) < 0 for all xi beyond cutoff (parasitic regime)."""
        xi = np.linspace(0, np.pi, self.N_XI)
        for p in [1, 2, 3, 4]:
            profile = interior_group_velocity(p=p, nu=1, xi_array=xi)
            beyond = xi > profile.cutoff_xi
            assert np.any(beyond), f"E{2*p}: no points beyond cutoff"
            C_beyond = profile.group_velocity[beyond]
            assert np.all(C_beyond <= 0), (
                f"E{2*p}: found positive C beyond cutoff, "
                f"max = {np.max(C_beyond):.4e}"
            )

class TestBoundaryGroupVelocity:
    """Boundary closure group velocity analysis (34.3)."""

    N_XI = 1000

    def test_boundary_gv_returns_all_rows(self):
        """boundary_group_velocity returns a profile for each boundary row."""
        xi = np.linspace(0, np.pi, self.N_XI)
        # E2, q=1, nextra=0, tension kernel with small sigma
        profiles = boundary_group_velocity(
            p=1, q=1, nextra=0, nu=1, sigma=0.1, kernel="tension", xi_array=xi,
        )
        # For E2 nu=1: r = q+1+nextra = 2
        assert len(profiles) == 2
        for i in range(2):
            assert i in profiles
            assert isinstance(profiles[i], GroupVelocityProfile)
            assert len(profiles[i].xi) == self.N_XI
            assert profiles[i].order == 1  # boundary accuracy order = q

    def test_boundary_gv_bounded(self):
        """For E2/E4 at small sigma, |C(xi)| is bounded (no blow-up)."""
        xi = np.linspace(0.01, np.pi - 0.01, self.N_XI)
        configs = [
            (1, 1, 0, 0.1),  # E2, q=1
            (2, 3, 0, 0.1),  # E4, q=3
        ]
        for p, q, nextra, sigma in configs:
            profiles = boundary_group_velocity(
                p=p, q=q, nextra=nextra, nu=1, sigma=sigma,
                kernel="tension", xi_array=xi,
            )
            for i, prof in profiles.items():
                C = prof.group_velocity
                assert np.all(np.isfinite(C)), (
                    f"p={p}, row {i}: non-finite group velocity"
                )
                assert np.max(np.abs(C)) < 100, (
                    f"p={p}, row {i}: |C| blow-up, max={np.max(np.abs(C)):.2e}"
                )

    def test_boundary_row0_low_xi_near_unity(self):
        """Boundary row 0 group velocity should approach 1 at low xi (consistent scheme)."""
        xi = np.linspace(0.01, 0.3, self.N_XI)
        profiles = boundary_group_velocity(
            p=2, q=3, nextra=0, nu=1, sigma=0.1, kernel="tension", xi_array=xi,
        )
        # Row 0 evaluates derivative at grid point 0 using a one-sided stencil.
        # At low xi, if the scheme is consistent, C should be near 1.
        C = profiles[0].group_velocity
        assert abs(C[0] - 1.0) < 0.5, (
            f"Boundary row 0 C at xi={xi[0]:.3f} = {C[0]:.4f}, expected ~1"
        )

    def test_cutoff_handles_oscillating_c(self):
        """cutoff_xi reflects persistent (not transient) sign reversal.

        Boundary stencils can have C(xi) that dips below zero briefly
        then recovers positive.  The cutoff should mark where C goes
        *permanently* non-positive, not the first transient dip.
        """
        from stencil_gen.group_velocity import _build_profile

        # Synthetic one-sided stencil (i_eval=0, nodes=[0..4]) whose
        # group velocity oscillates: C = 3cos(xi) - 6cos(2xi) + 6cos(3xi) - 2cos(4xi).
        # C(0) = 1, dips negative near xi~0.7, recovers strongly positive
        # around xi~pi/2, then goes permanently negative near xi~2.7.
        weights = [0.0, 3.0, -3.0, 2.0, -0.5]
        nodes = [0, 1, 2, 3, 4]
        xi = np.linspace(0, np.pi, 2000)

        profile = _build_profile(weights, np.asarray(nodes), xi, order=1)
        C = profile.group_velocity

        # Verify oscillation: C has a transient dip below zero AND
        # later recovery to positive values before the final descent.
        first_neg = None
        for idx in range(1, len(xi)):
            if C[idx] <= 0.0:
                first_neg = float(xi[idx])
                break
        recovery = False
        if first_neg is not None:
            for idx in range(1, len(xi)):
                if xi[idx] > first_neg and C[idx] > 0.0:
                    recovery = True
                    break
        assert first_neg is not None, "expected a transient negative dip"
        assert recovery, "expected C to recover positive after first dip"

        # Cutoff must be beyond the transient dip (at the persistent crossing)
        assert profile.cutoff_xi > first_neg + 0.5, (
            f"cutoff_xi={profile.cutoff_xi:.3f} is too close to first "
            f"transient dip at xi={first_neg:.3f}"
        )

        # Beyond cutoff, C stays non-positive
        beyond = xi > profile.cutoff_xi
        if np.any(beyond):
            assert np.all(C[beyond] <= 0.0), (
                f"C has positive values beyond cutoff_xi={profile.cutoff_xi:.3f}"
            )

    def test_boundary_vs_interior_gv_error(self):
        """Boundary row 0 has larger GV error than interior; no sign reversal at low xi.

        Row 0 (fully one-sided) should have larger error than the symmetric
        interior stencil.  Rows closer to the interior may have smaller error
        because they use a wider stencil.

        At well-resolved wavenumbers (xi < pi/4), no boundary row should
        reverse the sign of C (which would mean energy propagating backwards).
        """
        xi = np.linspace(0.01, np.pi, self.N_XI)
        for p, q, nextra, sigma in [(2, 3, 0, 0.1), (3, 5, 0, 0.1)]:
            interior = interior_group_velocity(p=p, nu=1, xi_array=xi)
            boundary = boundary_group_velocity(
                p=p, q=q, nextra=nextra, nu=1, sigma=sigma,
                kernel="tension", xi_array=xi,
            )
            resolved = xi < np.pi / 4
            # Row 0 (fully one-sided) should have the largest error
            C_row0 = boundary[0].group_velocity[resolved]
            C_int = interior.group_velocity[resolved]
            row0_err = np.max(np.abs(C_row0 - 1.0))
            int_err = np.max(np.abs(C_int - 1.0))
            assert row0_err >= int_err, (
                f"p={p}: row 0 error ({row0_err:.4e}) smaller than "
                f"interior ({int_err:.4e})"
            )
            # No sign reversal at well-resolved wavenumbers for any row
            for i, prof in boundary.items():
                C_bnd = prof.group_velocity[resolved]
                assert np.all(C_bnd > 0), (
                    f"p={p}, row {i}: C < 0 at resolved xi<pi/4 "
                    f"(min={np.min(C_bnd):.4e})"
                )

    def test_parasitic_direction_at_boundary(self):
        """Check for parasitic outgoing modes at the boundary.

        For u_t + u_x = 0 on a left boundary, physical waves propagate rightward
        (C > 0).  If a boundary stencil creates a mode with C < 0 at a wavenumber
        where the interior has C > 0, that's a parasitic mode propagating energy
        out through the boundary — the hallmark of GKS instability.

        Here we check the complementary concern for an inflow boundary: modes
        with C > 0 (into the domain) at wavenumbers where the interior has C < 0
        (energy should be propagating outward).  Such modes create spontaneous
        radiation of energy into the domain.
        """
        xi = np.linspace(0.01, np.pi - 0.01, self.N_XI)
        for p, q, nextra, sigma in [(1, 1, 0, 0.1), (2, 3, 0, 0.1)]:
            interior = interior_group_velocity(p=p, nu=1, xi_array=xi)
            boundary = boundary_group_velocity(
                p=p, q=q, nextra=nextra, nu=1, sigma=sigma,
                kernel="tension", xi_array=xi,
            )
            # In the parasitic regime (beyond interior cutoff), check if
            # boundary rows create C > 0 where interior has C < 0.
            parasitic = xi > interior.cutoff_xi
            if not np.any(parasitic):
                continue
            for i, prof in boundary.items():
                C_bnd_parasitic = prof.group_velocity[parasitic]
                # Flag if boundary creates strongly positive C in parasitic regime.
                # Small positive values are tolerable; large positive means energy
                # radiation into the domain.
                max_positive = np.max(C_bnd_parasitic) if len(C_bnd_parasitic) > 0 else 0
                # This is a diagnostic — we record but don't fail on small positives.
                # A strongly positive boundary C in the parasitic regime is suspicious.
                assert max_positive < 5.0, (
                    f"p={p}, row {i}: boundary C strongly positive ({max_positive:.2f}) "
                    f"in parasitic regime — potential GKS instability source"
                )


class TestBoundaryClassical:
    """Classical (non-RBF) boundary stencil group velocity analysis (34.3b)."""

    N_XI = 1000

    @pytest.fixture(scope="class")
    def e4_classical(self):
        """Derive E4 boundary rows with conservation and known-good alpha values."""
        from sympy import symbols

        from stencil_gen.boundary import derive_boundary
        from stencil_gen.conservation import build_conservation_system, solve_conservation

        result = derive_boundary(p=2, nu=1, s=0)
        equations, w_syms, last_free = build_conservation_system(
            result.r, result.t, 2, result.rows, result.interior_coeffs,
        )
        _, updated_rows = solve_conservation(
            equations, w_syms, last_free, result.all_free_params, result.rows,
        )
        # Known-good alpha values from E4u_1.t.cpp (same as test_boundary.py)
        a0, a1 = symbols("alpha_0 alpha_1")
        alpha_values = {a0: -0.7733323791884821, a1: 0.1623961700641681}
        return updated_rows, alpha_values

    def test_classical_returns_all_rows(self, e4_classical):
        """boundary_group_velocity_classical returns a profile for each boundary row."""
        updated_rows, alpha_values = e4_classical
        xi = np.linspace(0, np.pi, self.N_XI)
        profiles = boundary_group_velocity_classical(
            updated_rows, alpha_values, order=3, xi_array=xi,
        )
        # E4 (p=2): r = 3 boundary rows
        assert len(profiles) == 3
        for i in range(3):
            assert i in profiles
            assert isinstance(profiles[i], GroupVelocityProfile)
            assert profiles[i].order == 3

    def test_classical_coefficients_finite(self, e4_classical):
        """All evaluated coefficients are finite (no unresolved symbols)."""
        updated_rows, alpha_values = e4_classical
        xi = np.linspace(0.01, np.pi - 0.01, self.N_XI)
        profiles = boundary_group_velocity_classical(
            updated_rows, alpha_values, order=3, xi_array=xi,
        )
        for i, prof in profiles.items():
            assert np.all(np.isfinite(prof.group_velocity)), (
                f"Row {i}: non-finite group velocity"
            )
            assert np.all(np.isfinite(prof.kappa_star)), (
                f"Row {i}: non-finite modified wavenumber"
            )

    def test_classical_row0_low_xi(self, e4_classical):
        """Classical E4 row 0 group velocity near unity at low xi."""
        updated_rows, alpha_values = e4_classical
        xi = np.linspace(0.01, 0.3, self.N_XI)
        profiles = boundary_group_velocity_classical(
            updated_rows, alpha_values, order=3, xi_array=xi,
        )
        C = profiles[0].group_velocity
        assert abs(C[0] - 1.0) < 0.5, (
            f"Classical E4 row 0 C at xi={xi[0]:.3f} = {C[0]:.4f}, expected ~1"
        )

    def test_classical_bounded(self, e4_classical):
        """Group velocity is bounded (no blow-up) for all boundary rows."""
        updated_rows, alpha_values = e4_classical
        xi = np.linspace(0.01, np.pi - 0.01, self.N_XI)
        profiles = boundary_group_velocity_classical(
            updated_rows, alpha_values, order=3, xi_array=xi,
        )
        for i, prof in profiles.items():
            assert np.max(np.abs(prof.group_velocity)) < 100, (
                f"Row {i}: |C| blow-up, max={np.max(np.abs(prof.group_velocity)):.2e}"
            )

    def test_classical_e4_boundary_gv(self, e4_classical):
        """Classical E4 boundary stencils: no sign reversal at resolved wavenumbers.

        Uses the known-good alpha values from E4u_1.t.cpp.  At well-resolved
        wavenumbers (xi < pi/2), all boundary rows should have C > 0, meaning
        no parasitic energy reversal in the resolved regime.
        """
        updated_rows, alpha_values = e4_classical
        xi = np.linspace(0.01, np.pi, self.N_XI)
        profiles = boundary_group_velocity_classical(
            updated_rows, alpha_values, order=3, xi_array=xi,
        )
        resolved = xi < np.pi / 2
        for i, prof in profiles.items():
            C_resolved = prof.group_velocity[resolved]
            assert np.all(C_resolved > 0), (
                f"Classical E4 row {i}: C < 0 at resolved xi "
                f"(min={np.min(C_resolved):.4e}), parasitic sign reversal"
            )


class TestGKSDiagnostic:
    """GKS group velocity diagnostic tests (34.3e).

    Tests for :func:`gks_group_velocity_check`, which bridges per-stencil
    group velocity analysis with full-operator eigenvalue analysis by
    identifying boundary-localized, nearly-neutral eigenmodes whose group
    velocity radiates energy into the domain (GKS instability signature).
    """

    N_XI = 1000

    def test_stable_scheme_no_outgoing_modes(self):
        """Stable E2 scheme: no boundary modes radiate energy into the domain.

        E2 with tension kernel (universally stable for all sigma) produces no
        boundary-localized nearly-neutral eigenmodes at sigma=10, confirming
        the diagnostic returns clean results for a well-behaved scheme.
        """
        from stencil_gen.phs import build_diff_matrix_rbf

        n = 40
        xi = np.linspace(0, np.pi, self.N_XI)
        D = build_diff_matrix_rbf(
            n, p=1, q=1, epsilon=10.0, kernel="tension", nu=1, nextra=0,
        )
        modes = gks_group_velocity_check(D, xi)

        outgoing = [m for m in modes if m.is_outgoing]
        assert len(outgoing) == 0, (
            f"Stable E2 has {len(outgoing)} outgoing modes: "
            + "; ".join(
                f"lam={m.eigenvalue:.4f}, xi={m.boundary_wavenumber:.3f}, "
                f"C={m.group_velocity:.3f}"
                for m in outgoing
            )
        )

    def test_known_unstable_extrapolation(self):
        """GKS diagnostic detects parasitic boundary mode in E4 PHS.

        The original plan aimed to test with extrapolation outflow BC (known
        GKS-unstable for leapfrog, Trefethen 1983).  However, the leapfrog
        instability is time-discrete: the fully-discrete dispersion relation
        differs from the semi-discrete one, and the semi-discrete eigenvalue
        framework used here cannot capture that effect.  A time-discrete
        extension using the leapfrog dispersion relation is future work.

        Instead, we test with E4 PHS (sigma=0), which reliably produces a
        boundary-localized, nearly-neutral eigenmode whose dominant wavenumber
        is in the parasitic regime (xi near pi).  The interior group velocity
        at that wavenumber is strongly negative (C ~ -1.7), meaning energy
        flows from the right boundary into the domain.  The mode is slightly
        damped (Re ~ -0.007), so it doesn't cause exponential growth, but the
        diagnostic correctly flags the suspicious group velocity direction.
        """
        from stencil_gen.phs import build_diff_matrix_rbf

        n = 40
        xi = np.linspace(0, np.pi, self.N_XI)
        D = build_diff_matrix_rbf(
            n, p=2, q=3, epsilon=0.0, kernel="tension", nu=1, nextra=0,
        )
        modes = gks_group_velocity_check(D, xi)

        # E4 PHS produces at least one outgoing boundary mode
        outgoing = [m for m in modes if m.is_outgoing]
        assert len(outgoing) >= 1, (
            f"Expected at least 1 outgoing mode for E4 PHS, got {len(modes)} "
            f"total modes with 0 outgoing"
        )

        # The outgoing mode should be in the parasitic regime (high xi)
        # where interior group velocity is negative
        mode = outgoing[0]
        assert isinstance(mode, GKSModeInfo)
        assert mode.boundary_wavenumber > np.pi / 2, (
            f"Outgoing mode wavenumber {mode.boundary_wavenumber:.3f} should be "
            f"in parasitic regime (> pi/2)"
        )
        assert mode.group_velocity < 0, (
            f"Outgoing mode group velocity {mode.group_velocity:.3f} should be "
            f"negative (energy flowing into domain from right boundary)"
        )
        # Mode is nearly-neutral (slightly damped, not growing)
        assert mode.eigenvalue.real < 0, (
            f"Mode Re(lambda) = {mode.eigenvalue.real:.6e} should be negative "
            f"(damped, not growing)"
        )


class TestGKSSideParameter:
    """Tests for the ``side`` parameter of :func:`gks_group_velocity_check` (41.4b)."""

    N_XI = 1000

    def _build_e4_phs_matrix(self, n: int = 40) -> np.ndarray:
        """Build E4 PHS differentiation matrix (known to have boundary modes)."""
        from stencil_gen.phs import build_diff_matrix_rbf

        return build_diff_matrix_rbf(
            n, p=2, q=3, epsilon=0.0, kernel="tension", nu=1, nextra=0,
        )

    def test_left_default_unchanged(self):
        """Explicit side='left' produces the same results as the default (no side arg).

        Verifies backwards compatibility: adding the side parameter does not
        change existing behavior for the left boundary.
        """
        xi = np.linspace(0, np.pi, self.N_XI)
        D = self._build_e4_phs_matrix()

        modes_default = gks_group_velocity_check(D, xi)
        modes_left = gks_group_velocity_check(D, xi, side="left")

        assert len(modes_default) == len(modes_left)
        for m_def, m_left in zip(modes_default, modes_left):
            np.testing.assert_allclose(
                m_def.eigenvalue, m_left.eigenvalue, atol=1e-12,
            )
            np.testing.assert_allclose(
                m_def.boundary_wavenumber, m_left.boundary_wavenumber, atol=1e-12,
            )
            np.testing.assert_allclose(
                m_def.group_velocity, m_left.group_velocity, atol=1e-12,
            )
            assert m_def.is_outgoing == m_left.is_outgoing

    def test_right_mirrors_left(self):
        """side='right' on D has the same eigenvalue spectrum as side='left' on P@D@P.

        Under index reversal, ``D_rev = P @ D @ P`` (where ``P`` is the
        reversal permutation) satisfies ``D_rev[1:, 1:] = P' @ D[:-1, :-1] @ P'``
        (a similarity transform), so ``eig(-D_rev[1:, 1:]) == eig(-D[:-1, :-1])``.
        This means ``side="left"`` on ``D_rev`` produces the same eigenvalues as
        ``side="right"`` on the original ``D``.

        The outgoing classification is not compared because it depends on the
        eigenvector spatial structure, which the FFT-based wavenumber estimator
        resolves differently on the reversed vs original matrix.
        """
        xi = np.linspace(0, np.pi, self.N_XI)
        D = self._build_e4_phs_matrix()
        n = D.shape[0]

        # Index reversal: D_rev = P @ D @ P (no negation — preserves eig(-D_bc))
        P = np.eye(n)[::-1]
        D_rev = P @ D @ P

        modes_right = gks_group_velocity_check(D, xi, side="right")
        modes_left_rev = gks_group_velocity_check(D_rev, xi, side="left")

        # Both analyses should find the same number of modes
        assert len(modes_right) == len(modes_left_rev), (
            f"side='right' found {len(modes_right)} modes, "
            f"side='left' on reflected D found {len(modes_left_rev)}"
        )

        # Sort by eigenvalue imaginary part for stable comparison
        modes_right_sorted = sorted(modes_right, key=lambda m: m.eigenvalue.imag)
        modes_left_sorted = sorted(modes_left_rev, key=lambda m: m.eigenvalue.imag)

        for m_right, m_left in zip(modes_right_sorted, modes_left_sorted):
            # Eigenvalues should match (similarity transform preserves spectrum)
            np.testing.assert_allclose(
                m_right.eigenvalue, m_left.eigenvalue, atol=1e-8,
                err_msg="Eigenvalues should match between right and reflected-left",
            )

    def test_bottom_raises(self):
        """side='bottom' raises NotImplementedError (requires 2D D, deferred to 41.6)."""
        xi = np.linspace(0, np.pi, self.N_XI)
        D = self._build_e4_phs_matrix()

        with pytest.raises(NotImplementedError, match="2D differentiation matrix"):
            gks_group_velocity_check(D, xi, side="bottom")

    def test_top_raises(self):
        """side='top' raises NotImplementedError."""
        xi = np.linspace(0, np.pi, self.N_XI)
        D = self._build_e4_phs_matrix()

        with pytest.raises(NotImplementedError, match="2D differentiation matrix"):
            gks_group_velocity_check(D, xi, side="top")

    def test_invalid_side_raises(self):
        """Invalid side value raises ValueError."""
        xi = np.linspace(0, np.pi, self.N_XI)
        D = self._build_e4_phs_matrix()

        with pytest.raises(ValueError, match="side must be one of"):
            gks_group_velocity_check(D, xi, side="invalid")


class TestNonuniformModWavenumber:
    """Tests for modified_wavenumber_nonuniform and group_velocity_exact_nonuniform (35.1b)."""

    N_XI = 500

    def test_nonuniform_mod_wavenumber(self):
        """With integer offsets, nonuniform version matches uniform version exactly."""
        weights = [-0.5, 0.0, 0.5]
        nodes = [-1, 0, 1]
        i_eval = 0
        offsets = [n - i_eval for n in nodes]  # [-1, 0, 1]
        xi = np.linspace(0, np.pi, self.N_XI)

        kstar_uniform = modified_wavenumber(weights, i_eval, nodes, xi)
        kstar_nonuniform = modified_wavenumber_nonuniform(weights, offsets, xi)

        assert np.allclose(kstar_uniform, kstar_nonuniform, atol=1e-14), (
            f"Nonuniform should match uniform for integer offsets, "
            f"max diff = {np.max(np.abs(kstar_uniform - kstar_nonuniform)):.2e}"
        )

    def test_nonuniform_gv_matches_uniform(self):
        """With integer offsets, nonuniform GV matches uniform GV exactly."""
        weights = [-0.5, 0.0, 0.5]
        nodes = [-1, 0, 1]
        i_eval = 0
        offsets = [n - i_eval for n in nodes]
        xi = np.linspace(0, np.pi, self.N_XI)

        C_uniform = group_velocity_exact(weights, i_eval, nodes, xi)
        C_nonuniform = group_velocity_exact_nonuniform(weights, offsets, xi)

        assert np.allclose(C_uniform, C_nonuniform, atol=1e-14), (
            f"Nonuniform GV should match uniform for integer offsets, "
            f"max diff = {np.max(np.abs(C_uniform - C_nonuniform)):.2e}"
        )

    def test_nonuniform_fractional_offset_bounded(self):
        """Nonuniform with fractional offsets produces bounded, finite results."""
        # Simulate a cut-cell-like stencil with wall at -0.3
        weights = [-0.3, 0.7, 0.1, -0.5]
        offsets = [-0.3, 0.0, 1.0, 2.0]
        xi = np.linspace(0.01, np.pi, self.N_XI)

        kstar = modified_wavenumber_nonuniform(weights, offsets, xi)
        C = group_velocity_exact_nonuniform(weights, offsets, xi)

        assert np.all(np.isfinite(kstar)), "kappa* should be finite"
        assert np.all(np.isfinite(C)), "Group velocity should be finite"
        assert np.max(np.abs(C)) < 100, f"|C| blow-up: max={np.max(np.abs(C)):.2e}"


class TestCutCellGroupVelocity:
    """Cut-cell group velocity analysis (35.1c)."""

    N_XI = 1000

    @pytest.fixture(scope="class")
    def e2_1_cut_cell(self):
        """Derive E2_1 cut-cell stencil (symbolic in psi and alpha)."""
        from sympy import Symbol

        from stencil_gen.temo import E2_1, derive_cut_cell_scheme

        psi = Symbol("psi")
        result = derive_cut_cell_scheme(E2_1, psi)
        return result, psi

    def test_psi_1_matches_uniform(self, e2_1_cut_cell):
        """At psi=1, cut-cell group velocity matches uniform boundary GV.

        By TEMO construction, B(psi=1) = B_u (the uniform boundary stencil).
        At psi=1 the wall is at position -1 (uniform spacing), so the cut-cell
        stencil effectively reduces to a uniform-grid boundary stencil.

        We compare by evaluating the TEMO's own B_u at alpha=0 and verifying
        the group velocity profiles agree at resolved wavenumbers.
        """
        from stencil_gen.group_velocity import _build_profile
        from stencil_gen.temo import E2_1 as scheme, derive_uniform_boundary_for_temo

        result, psi = e2_1_cut_cell
        xi = np.linspace(0.01, np.pi, self.N_XI)
        alpha_vals = {s: 0 for s in result.alpha_symbols}

        cc_profiles = cut_cell_group_velocity(
            result, psi, psi_val=1.0, alpha_values=alpha_vals, xi_array=xi,
        )

        # At psi=1, all rows should give well-behaved GV
        for i, prof in cc_profiles.items():
            assert np.all(np.isfinite(prof.group_velocity)), (
                f"Row {i}: non-finite GV at psi=1"
            )
            # At low xi, C should be near 1 (consistent scheme)
            low_xi = xi < 0.3
            C_low = prof.group_velocity[low_xi]
            assert abs(C_low[0] - 1.0) < 0.5, (
                f"Row {i}: C at low xi = {C_low[0]:.4f}, expected ~1"
            )

        # Get the TEMO uniform boundary B_u and compute its GV.
        # At psi=1, the wall column coefficient is zero for rows 0..r-1,
        # so the cut-cell stencil exactly matches the uniform one.
        ur = derive_uniform_boundary_for_temo(scheme)
        B_u = ur.B_u
        u_alpha_vals = {s: 0 for s in ur.alpha_symbols}
        t = B_u.cols
        nodes = np.arange(t)

        for i in range(B_u.rows):
            w_uni = [float(B_u[i, j].xreplace(u_alpha_vals))
                     if hasattr(B_u[i, j], 'xreplace') else float(B_u[i, j])
                     for j in range(t)]
            uni_prof = _build_profile(w_uni, nodes - i, xi, order=scheme.q)
            max_diff = np.max(np.abs(
                cc_profiles[i].group_velocity - uni_prof.group_velocity
            ))
            assert max_diff < 1e-10, (
                f"Row {i}: psi=1 cut-cell GV should match TEMO uniform "
                f"exactly (wall coeff=0), diff = {max_diff:.2e}"
            )

    def test_psi_0_degenerate_bounded(self, e2_1_cut_cell):
        """At psi=0 (degenerate mesh), group velocity is bounded and finite.

        The degenerate point collocates the wall with grid point 0, so the
        stencil degenerates gracefully (by TEMO design principle).
        """
        result, psi = e2_1_cut_cell
        xi = np.linspace(0.01, np.pi - 0.01, self.N_XI)
        alpha_vals = {s: 0 for s in result.alpha_symbols}

        profiles = cut_cell_group_velocity(
            result, psi, psi_val=0.0, alpha_values=alpha_vals, xi_array=xi,
        )

        for i, prof in profiles.items():
            C = prof.group_velocity
            assert np.all(np.isfinite(C)), (
                f"Row {i}: non-finite GV at psi=0"
            )
            assert np.max(np.abs(C)) < 100, (
                f"Row {i}: |C| blow-up at psi=0, max={np.max(np.abs(C)):.2e}"
            )

    def test_e2_1_cut_cell_gv_smooth_in_psi(self, e2_1_cut_cell):
        """E2_1 group velocity varies smoothly with psi (no discontinuous jumps).

        Compute C(xi) at 11 evenly spaced psi values in [0, 1] and verify
        that the group velocity profile at resolved wavenumbers changes
        smoothly between adjacent values (bounded derivative dC/dpsi).
        """
        result, psi = e2_1_cut_cell
        xi = np.linspace(0.01, np.pi / 2, self.N_XI)
        alpha_vals = {s: 0 for s in result.alpha_symbols}

        psi_values = np.linspace(0.0, 1.0, 11)
        all_profiles = {}
        for pv in psi_values:
            all_profiles[pv] = cut_cell_group_velocity(
                result, psi, psi_val=float(pv), alpha_values=alpha_vals,
                xi_array=xi,
            )

        # For each row, check that adjacent psi values give bounded dC/dpsi
        R = result.floating.rows
        for i in range(R):
            for k in range(len(psi_values) - 1):
                pv0, pv1 = float(psi_values[k]), float(psi_values[k + 1])
                C0 = all_profiles[pv0][i].group_velocity
                C1 = all_profiles[pv1][i].group_velocity
                dpsi = pv1 - pv0
                max_deriv = np.max(np.abs(C1 - C0)) / dpsi
                # dC/dpsi should be finite (no discontinuity).  A smooth
                # rational function in psi can have large but bounded
                # derivatives, especially near psi=0 where the stencil
                # degenerates.  Threshold of 200 catches genuine
                # discontinuities while allowing smooth variation.
                assert max_deriv < 200, (
                    f"Row {i}: dC/dpsi too large between psi={pv0:.2f} and "
                    f"psi={pv1:.2f}, max|dC/dpsi| = {max_deriv:.1f}"
                )


class TestPsiSweepGroupVelocity:
    """Psi sweep group velocity analysis (35.2b)."""

    N_XI = 500

    def test_e2_1_psi_sweep(self):
        """Sweep psi in [0, 1] for E2_1; verify result structure and diagnostics.

        Computes group velocity at 11 psi values and verifies:
        - PsiSweepResult is well-formed with all expected fields.
        - All profiles are finite and bounded.
        - No parasitic sign reversal at well-resolved wavenumbers (xi < pi/2)
          for non-degenerate psi values (psi >= 0.1).
        """
        from stencil_gen.temo import E2_1

        xi = np.linspace(0.01, np.pi, self.N_XI)
        psi_values = np.linspace(0.0, 1.0, 11)

        result = psi_sweep_group_velocity(
            E2_1, psi_values, alpha_values={}, xi_array=xi,
        )

        assert isinstance(result, PsiSweepResult)
        assert len(result.profiles) == len(psi_values)
        assert result.min_C < float("inf")

        # All profiles should be finite and bounded
        for pv, profs in result.profiles.items():
            for row_idx, prof in profs.items():
                assert np.all(np.isfinite(prof.group_velocity)), (
                    f"psi={pv}, row {row_idx}: non-finite GV"
                )
                assert np.max(np.abs(prof.group_velocity)) < 100, (
                    f"psi={pv}, row {row_idx}: |C| blow-up"
                )

        # At non-degenerate psi (>= 0.1), boundary rows should not have
        # strongly negative C at resolved wavenumbers.  The degenerate
        # psi=0 point collocates the wall with a grid point and some rows
        # naturally produce negative C there.
        resolved = xi < np.pi / 2
        interior = interior_group_velocity(p=E2_1.p, nu=1, xi_array=xi)
        C_int = interior.group_velocity
        for pv, profs in result.profiles.items():
            if pv < 0.1:
                continue
            for row_idx, prof in profs.items():
                C_res = prof.group_velocity[resolved]
                # No parasitic sign reversal: boundary C > 0 where interior C < 0
                reversal = (C_res > 0) & (C_int[resolved] < 0)
                assert not np.any(reversal), (
                    f"psi={pv}, row {row_idx}: parasitic sign reversal "
                    f"at resolved xi"
                )

    def test_e2_1_no_cfl_penalty(self):
        """TEMO cut-cell stencil does not dramatically increase max|omega(xi)|.

        The TEMO construction avoids the CFL stiffness penalty: the maximum
        |omega| = max|Im(kappa*(xi))| should not blow up as psi -> 0.
        We verify that the ratio max|omega(psi)| / max|omega(psi=1)| stays
        bounded (< 10x) across all psi values.
        """
        from stencil_gen.temo import E2_1

        xi = np.linspace(0.01, np.pi, self.N_XI)
        psi_values = np.linspace(0.0, 1.0, 11)

        result = psi_sweep_group_velocity(
            E2_1, psi_values, alpha_values={}, xi_array=xi,
        )

        # Compute max|omega| at psi=1 as reference
        ref_profs = result.profiles[1.0]
        ref_max_omega = max(
            float(np.max(np.abs(np.imag(prof.kappa_star))))
            for prof in ref_profs.values()
        )

        # At each psi, max|omega| should not blow up
        for pv, profs in result.profiles.items():
            psi_max_omega = max(
                float(np.max(np.abs(np.imag(prof.kappa_star))))
                for prof in profs.values()
            )
            ratio = psi_max_omega / max(ref_max_omega, 1e-15)
            assert ratio < 10.0, (
                f"psi={pv}: max|omega| = {psi_max_omega:.4f} is "
                f"{ratio:.1f}x the psi=1 reference ({ref_max_omega:.4f}), "
                f"indicating CFL penalty"
            )

    @pytest.mark.slow
    def test_e4_1_psi_sweep(self):
        """Sweep psi in [0, 1] for E4_1 (stricter scheme).

        E4_1 has tighter constraints (only 2 stable schemes in the paper).
        Verify the psi sweep completes without blow-up and profiles are bounded.
        """
        from stencil_gen.temo import E4_1

        xi = np.linspace(0.01, np.pi, self.N_XI)
        psi_values = np.linspace(0.0, 1.0, 11)

        result = psi_sweep_group_velocity(
            E4_1, psi_values, alpha_values={}, xi_array=xi,
        )

        assert isinstance(result, PsiSweepResult)
        assert len(result.profiles) == len(psi_values)

        # All profiles should be finite and bounded
        for pv, profs in result.profiles.items():
            for row_idx, prof in profs.items():
                assert np.all(np.isfinite(prof.group_velocity)), (
                    f"psi={pv}, row {row_idx}: non-finite GV"
                )
                # E4_1 has a wider stencil (7 points) with non-uniform
                # offsets, so |C| can be larger at high xi than for E2.
                assert np.max(np.abs(prof.group_velocity)) < 500, (
                    f"psi={pv}, row {row_idx}: |C| blow-up, "
                    f"max={np.max(np.abs(prof.group_velocity)):.2e}"
                )


class TestCutCellGVvsEigenvalue:
    """Comparison of GV diagnostic with eigenvalue analysis (35.3a).

    Verifies that the per-stencil group velocity diagnostic agrees with
    full-operator eigenvalue stability for cut-cell configurations, and
    demonstrates the O(1)-vs-O(N^3) cost advantage of the GV approach.
    """

    N_XI = 500

    @staticmethod
    def _build_cut_cell_diff_matrix(cc_result, psi_sym, psi_val, alpha_values,
                                    scheme_params, n):
        """Build N×N diff matrix with cut-cell left boundary.

        Left boundary rows use the Dirichlet cut-cell stencil evaluated at
        *psi_val*.  Interior uses the standard centered stencil.  Right
        boundary uses the uniform RBF/tension stencil (sigma=10, stable).

        Parameters
        ----------
        cc_result : CutCellResult
            Pre-derived symbolic cut-cell stencil.
        psi_sym : Symbol
            SymPy psi symbol.
        psi_val : float
            Numeric psi value.
        alpha_values : dict
            Alpha symbol → value mapping.
        scheme_params : SchemeParams
            Scheme parameters (for p, q, nu, nextra).
        n : int
            Grid size.

        Returns
        -------
        np.ndarray
            N×N differentiation matrix.
        """
        from stencil_gen.interior import derive_interior, full_gamma_array
        from stencil_gen.phs import uniform_boundary_weights_rbf

        dims = cc_result.dims
        r, t, T = dims.r, dims.t, dims.T
        p, nu = scheme_params.p, scheme_params.nu

        subs = {psi_sym: psi_val, **alpha_values}

        # Interior stencil
        interior_coeffs = derive_interior(0, p, nu)
        interior_w = [float(c) for c in full_gamma_array(interior_coeffs)]

        D = np.zeros((n, n))

        # Left boundary: cut-cell Dirichlet stencil.
        # dirichlet has (R-1) = r rows, T columns.
        # Column 0 is the wall (u_wall = 0 for Dirichlet), columns 1..T-1
        # are grid points 0..T-2 = 0..t-1.
        F_dir = cc_result.dirichlet
        for i in range(F_dir.rows):
            for j in range(1, T):
                D[i, j - 1] = float(F_dir[i, j].xreplace(subs))

        # Interior rows: centered 2p+1 stencil
        for i in range(r, n - r):
            for k_idx, j in enumerate(range(i - p, i + p + 1)):
                D[i, j] = interior_w[k_idx]

        # Right boundary: reflected uniform stencil (sigma=0 tension =
        # polynomial-only, matching the TEMO polynomial design).
        sign = (-1.0) ** nu
        for i in range(r):
            w = uniform_boundary_weights_rbf(
                i, t, nu, scheme_params.q, 0.0, kernel="tension",
            )
            row = n - 1 - i
            for j in range(t):
                col = n - 1 - j
                D[row, col] = sign * float(w[j])

        return D

    def test_gv_predicts_eigenvalue_stability(self):
        """GV diagnostic and eigenvalue stability agree for E2_1 cut-cell.

        For E2_1 at psi = 0.1, 0.3, 0.5, 0.7, 1.0:
        - Compute GV profiles via cut_cell_group_velocity.
        - Build the N×N diff matrix and compute eigenvalues of -D.
        - Check: if no parasitic sign reversal in GV → eigenvalues stable
          (Re(lambda) <= small positive tolerance).
        """
        from sympy import Symbol

        from stencil_gen.temo import E2_1, derive_cut_cell_scheme

        psi_sym = Symbol("psi")
        cc = derive_cut_cell_scheme(E2_1, psi_sym)
        alpha_vals = {s: 0 for s in cc.alpha_symbols}

        xi = np.linspace(0.01, np.pi, self.N_XI)
        n = 40
        psi_test = [0.1, 0.3, 0.5, 0.7, 1.0]

        # Interior GV for sign-reversal detection
        interior = interior_group_velocity(p=E2_1.p, nu=1, xi_array=xi)
        C_int = interior.group_velocity

        for pv in psi_test:
            # --- Group velocity diagnostic ---
            profiles = cut_cell_group_velocity(
                cc, psi_sym, pv, alpha_vals, xi, order=E2_1.q,
            )
            # Check for parasitic sign reversal at resolved wavenumbers
            resolved = xi < np.pi / 2
            gv_has_reversal = False
            for row_idx, prof in profiles.items():
                C = prof.group_velocity
                reversal = (C[resolved] > 0) & (C_int[resolved] < 0)
                if np.any(reversal):
                    gv_has_reversal = True
                    break

            # --- Eigenvalue stability ---
            D = self._build_cut_cell_diff_matrix(
                cc, psi_sym, pv, alpha_vals, E2_1, n,
            )
            eigenvalues = np.linalg.eigvals(-D)
            max_real = float(np.max(eigenvalues.real))

            # Consistency: no GV reversal → eigenvalues stable.
            # Tolerance 1e-4: the polynomial-only right boundary produces
            # O(1e-6) positive Re at finite N — these vanish as N → ∞ and
            # are not related to the cut-cell left boundary under test.
            if not gv_has_reversal:
                assert max_real < 1e-4, (
                    f"psi={pv}: GV says no parasitic reversal but "
                    f"eigenvalue Re(lambda)_max = {max_real:.2e} > 0 "
                    f"(unstable)"
                )
            # Note: GV reversal does NOT guarantee instability (it's a
            # necessary condition from GKS theory, not sufficient), so we
            # don't assert the converse.

class Test2DGroupVelocity:
    """2D tensor-product group velocity tests (36.1)."""

    N_XI = 500

    @staticmethod
    def _e2_kappa_star(xi: np.ndarray) -> np.ndarray:
        """E2 (2nd-order central) modified wavenumber: kappa* = i*sin(xi)."""
        weights = [-0.5, 0.0, 0.5]
        nodes = [-1, 0, 1]
        return modified_wavenumber(weights, 0, nodes, xi)

    def test_2d_basic(self):
        """group_velocity_2d returns correct result structure and values for E2."""
        xi = np.linspace(0.01, np.pi - 0.01, self.N_XI)
        eta = np.linspace(0.01, np.pi - 0.01, self.N_XI)

        kx = self._e2_kappa_star(xi)
        ky = self._e2_kappa_star(eta)

        result = group_velocity_2d(kx, ky, xi, eta, a=1.0, b=1.0)

        # Check return type and field shapes
        assert isinstance(result, GroupVelocity2DResult)
        assert result.C_x.shape == (len(xi), len(eta))
        assert result.C_y.shape == (len(xi), len(eta))
        assert result.speed.shape == (len(xi), len(eta))
        assert result.angle.shape == (len(xi), len(eta))
        assert result.angle_error.shape == (len(xi), len(eta))

        # For E2, Im(kappa*) = sin(xi), so C_1d = cos(xi).
        # C_x should equal cos(xi) broadcast over eta.
        C_x_expected = np.cos(xi)
        np.testing.assert_allclose(
            result.C_x[:, len(eta) // 2], C_x_expected, atol=1e-3,
            err_msg="C_x should match cos(xi) for E2 stencil",
        )

        # C_y should equal cos(eta) broadcast over xi.
        C_y_expected = np.cos(eta)
        np.testing.assert_allclose(
            result.C_y[len(xi) // 2, :], C_y_expected, atol=1e-3,
            err_msg="C_y should match cos(eta) for E2 stencil",
        )

        # Speed at (xi, eta) = sqrt(cos^2(xi) + cos^2(eta))
        xi_2d, eta_2d = np.meshgrid(xi, eta, indexing="ij")
        speed_expected = np.sqrt(np.cos(xi_2d)**2 + np.cos(eta_2d)**2)
        np.testing.assert_allclose(
            result.speed, speed_expected, atol=1e-3,
            err_msg="Speed should be sqrt(C_x^2 + C_y^2)",
        )

    def test_2d_wave_speed_scaling(self):
        """group_velocity_2d correctly scales by wave speed coefficients a, b."""
        xi = np.linspace(0.1, 2.0, 200)
        eta = np.linspace(0.1, 2.0, 200)

        kx = self._e2_kappa_star(xi)
        ky = self._e2_kappa_star(eta)

        a, b = 2.0, 0.5
        result = group_velocity_2d(kx, ky, xi, eta, a=a, b=b)

        # C_x should be a*cos(xi), C_y should be b*cos(eta).
        # Exclude endpoints where np.gradient has reduced accuracy.
        s = slice(1, -1)
        np.testing.assert_allclose(
            result.C_x[s, 0], a * np.cos(xi[s]), atol=1e-3,
            err_msg="C_x should scale with wave speed a",
        )
        np.testing.assert_allclose(
            result.C_y[0, s], b * np.cos(eta[s]), atol=1e-3,
            err_msg="C_y should scale with wave speed b",
        )

    def test_anisotropy_basic(self):
        """anisotropy_profile returns correct structure and basic properties."""
        theta = np.linspace(0.01, np.pi / 2 - 0.01, 200)
        xi_mag = 1.0

        result = anisotropy_profile(p=1, nu=1, theta_array=theta, xi_mag=xi_mag)

        # Check return type and shapes
        assert isinstance(result, AnisotropyResult)
        assert result.C_x.shape == theta.shape
        assert result.C_y.shape == theta.shape
        assert result.speed.shape == theta.shape
        assert result.angle.shape == theta.shape
        assert result.angle_error.shape == theta.shape

        # For E2 (p=1), the 1D group velocity is cos(xi).
        # At theta=0: C_x = cos(xi_mag), C_y = 0, speed = cos(xi_mag)
        # At theta=pi/2: C_x = 0, C_y = cos(xi_mag), speed = cos(xi_mag)
        # At theta=pi/4: speed = cos(xi_mag/sqrt(2))
        # Since cos(xi_mag/sqrt(2)) > cos(xi_mag), diagonal is faster.

        # Verify speed is in (0, 1] for moderate xi_mag
        assert np.all(result.speed > 0), "Speed should be positive"
        assert np.all(result.speed <= 1.0 + 1e-10), "Speed should not exceed 1"

        # Angle should roughly track theta (anisotropy causes deviation,
        # up to ~0.18 rad for E2 at xi_mag=1.0 -- that's the physical effect)
        assert np.max(np.abs(result.angle_error)) < 0.25, (
            "Angle error should be bounded for moderate xi_mag"
        )

    def test_anisotropy_axis_vs_diagonal(self):
        """For E2, diagonal propagation is faster than axis-aligned (Trefethen)."""
        theta = np.array([0.01, np.pi / 4])  # near-axis and diagonal
        xi_mag = 1.0

        result = anisotropy_profile(p=1, nu=1, theta_array=theta, xi_mag=xi_mag)

        speed_axis = result.speed[0]      # theta ~ 0
        speed_diag = result.speed[1]      # theta = pi/4

        assert speed_diag > speed_axis, (
            f"E2 diagonal speed ({speed_diag:.6f}) should exceed "
            f"axis speed ({speed_axis:.6f})"
        )

    def test_anisotropy_small_xi_near_exact(self):
        """At small wavenumber, all schemes approach exact group velocity."""
        theta = np.linspace(0.1, np.pi / 2 - 0.1, 100)
        xi_mag = 0.1  # very resolved wave

        # Error is O(xi^(2p)) for order-2p scheme.  At xi=0.1:
        #   E2 (p=1): ~5e-3, E4 (p=2): ~5e-5, E6 (p=3): ~5e-7
        tols = {1: 1e-2, 2: 1e-4, 3: 1e-6}
        for p in [1, 2, 3]:  # E2, E4, E6
            result = anisotropy_profile(p=p, nu=1, theta_array=theta, xi_mag=xi_mag)
            np.testing.assert_allclose(
                result.speed, 1.0, atol=tols[p],
                err_msg=f"E{2*p} speed should be ~1 at xi_mag=0.1",
            )
            np.testing.assert_allclose(
                result.angle_error, 0.0, atol=tols[p],
                err_msg=f"E{2*p} angle error should be ~0 at xi_mag=0.1",
            )

    def test_axis_aligned_reduces_to_1d(self):
        """For theta=0 (wave along x-axis), 2D group velocity reduces to 1D."""
        xi = np.linspace(0.01, np.pi - 0.01, self.N_XI)
        eta = np.linspace(0.01, np.pi - 0.01, self.N_XI)

        kx = self._e2_kappa_star(xi)
        ky = self._e2_kappa_star(eta)

        # Advection purely in x: a=1, b=0
        result = group_velocity_2d(kx, ky, xi, eta, a=1.0, b=0.0)

        # C_x should match 1D group velocity cos(xi) for E2
        C_x_expected = np.cos(xi)
        np.testing.assert_allclose(
            result.C_x[:, len(eta) // 2], C_x_expected, atol=1e-3,
            err_msg="C_x should match 1D group velocity when b=0",
        )

        # C_y should be 0 everywhere since b=0
        np.testing.assert_allclose(
            result.C_y, 0.0, atol=1e-15,
            err_msg="C_y should be 0 for axis-aligned propagation (b=0)",
        )

        # Speed should equal |C_x| = |cos(xi)|
        np.testing.assert_allclose(
            result.speed[:, len(eta) // 2], np.abs(C_x_expected), atol=1e-3,
            err_msg="Speed should reduce to |C_x| = 1D speed when b=0",
        )

    def test_diagonal_propagation(self):
        """For theta=pi/4 (diagonal), C_x = C_y by symmetry."""
        # Use the anisotropy_profile which parameterizes by propagation angle
        theta = np.array([np.pi / 4])
        xi_mag = 1.0

        result = anisotropy_profile(p=1, nu=1, theta_array=theta, xi_mag=xi_mag)

        # At theta = pi/4: cos(theta) = sin(theta), and the 1D wavenumber
        # components are equal, so C_x == C_y
        np.testing.assert_allclose(
            result.C_x, result.C_y, atol=1e-14,
            err_msg="C_x should equal C_y at theta=pi/4 by symmetry",
        )

        # Group propagation angle should be exactly pi/4
        np.testing.assert_allclose(
            result.angle, np.pi / 4, atol=1e-14,
            err_msg="Group angle should be pi/4 for diagonal propagation",
        )

    def test_anisotropy_e4_reduced(self):
        """E4 has less grid anisotropy than E2 at the same wavenumber."""
        theta = np.linspace(0.01, np.pi / 2 - 0.01, 200)
        xi_mag = 1.0

        result_e2 = anisotropy_profile(p=1, nu=1, theta_array=theta, xi_mag=xi_mag)
        result_e4 = anisotropy_profile(p=2, nu=1, theta_array=theta, xi_mag=xi_mag)

        # Measure anisotropy as max-min speed variation over theta
        aniso_e2 = np.max(result_e2.speed) - np.min(result_e2.speed)
        aniso_e4 = np.max(result_e4.speed) - np.min(result_e4.speed)

        assert aniso_e4 < aniso_e2, (
            f"E4 anisotropy ({aniso_e4:.6f}) should be less than "
            f"E2 anisotropy ({aniso_e2:.6f})"
        )

        # Also check angle deviation is reduced
        max_angle_err_e2 = np.max(np.abs(result_e2.angle_error))
        max_angle_err_e4 = np.max(np.abs(result_e4.angle_error))

        assert max_angle_err_e4 < max_angle_err_e2, (
            f"E4 angle error ({max_angle_err_e4:.6f}) should be less than "
            f"E2 angle error ({max_angle_err_e2:.6f})"
        )

    def test_angle_deviation_bounded(self):
        """Group propagation angle deviation < 5 degrees for well-resolved waves.

        At xi_mag = 0.7 (~9 points per wavelength), E2 has ~4 deg deviation
        while E4 and E6 have much less.  All are bounded below 5 degrees.
        """
        theta = np.linspace(0.05, np.pi / 2 - 0.05, 200)
        xi_mag = 0.7  # ~9 points per wavelength

        for p in [1, 2, 3]:  # E2, E4, E6
            result = anisotropy_profile(
                p=p, nu=1, theta_array=theta, xi_mag=xi_mag,
            )

            max_deviation_rad = np.max(np.abs(result.angle_error))
            max_deviation_deg = np.degrees(max_deviation_rad)

            assert max_deviation_deg < 5.0, (
                f"E{2*p} angle deviation ({max_deviation_deg:.2f} deg) "
                f"exceeds 5 deg bound at xi_mag={xi_mag}"
            )

    def test_trefethen_eq_4_8a(self):
        """For E2, group speed matches the leading-order expansion.

        The tensor-product E2 group speed in 2D satisfies:
            |C| ~ 1 - |xi|^2 * (3 + cos(4*theta)) / 8
        to leading order in |xi|.  This is verified by Taylor-expanding
        cos(|xi|*cos(theta)) and cos(|xi|*sin(theta)) through the
        group velocity formula.
        """
        theta = np.linspace(0.05, np.pi / 2 - 0.05, 200)
        xi_mag = 0.15  # small enough for leading-order accuracy

        result = anisotropy_profile(p=1, nu=1, theta_array=theta, xi_mag=xi_mag)

        # Analytical prediction: |C| ~ 1 - xi^2*(3+cos(4*theta))/8
        predicted = 1 - xi_mag**2 * (3 + np.cos(4 * theta)) / 8

        # At xi=0.15, the next-order term is O(xi^4) ~ 5e-5.
        # The leading-order formula should match to ~1e-4.
        np.testing.assert_allclose(
            result.speed, predicted, atol=1e-4,
            err_msg="E2 group speed should match leading-order expansion",
        )

    def test_2d_boundary_basic(self):
        """boundary_group_velocity_2d returns per-row AnisotropyResult dicts."""
        xi = np.linspace(0, np.pi, self.N_XI)
        theta = np.linspace(0.05, np.pi / 2 - 0.05, 200)
        xi_mag = 0.7

        # Boundary stencils in x-direction (E2, q=1)
        bdy_x = boundary_group_velocity(
            p=1, q=1, nextra=0, nu=1, sigma=0.1,
            kernel="tension", xi_array=xi,
        )
        # Interior stencil in y-direction
        int_y = interior_group_velocity(p=1, nu=1, xi_array=xi)

        result = boundary_group_velocity_2d(bdy_x, int_y, theta, xi_mag)

        # Should return one AnisotropyResult per boundary row
        assert len(result) == len(bdy_x)
        for row_idx in bdy_x:
            assert row_idx in result
            r = result[row_idx]
            assert isinstance(r, AnisotropyResult)
            assert r.C_x.shape == theta.shape
            assert r.C_y.shape == theta.shape
            assert r.speed.shape == theta.shape

        # At moderate xi_mag, speed should be bounded and finite
        for row_idx, r in result.items():
            assert np.all(np.isfinite(r.speed)), (
                f"Row {row_idx}: non-finite speed"
            )
            assert np.all(r.speed < 10), (
                f"Row {row_idx}: speed blow-up, max={np.max(r.speed):.2f}"
            )

    def test_2d_boundary_interior_matches_anisotropy(self):
        """Far from boundary, 2D boundary GV should approach interior anisotropy."""
        xi = np.linspace(0, np.pi, 1000)
        theta = np.linspace(0.1, np.pi / 2 - 0.1, 100)
        xi_mag = 0.5

        # E4 boundary stencils -- the last (highest-index) boundary row
        # transitions toward the interior behavior
        bdy_x = boundary_group_velocity(
            p=2, q=3, nextra=0, nu=1, sigma=0.1,
            kernel="tension", xi_array=xi,
        )
        int_y = interior_group_velocity(p=2, nu=1, xi_array=xi)

        result_bdy = boundary_group_velocity_2d(bdy_x, int_y, theta, xi_mag)
        result_int = anisotropy_profile(p=2, nu=1, theta_array=theta, xi_mag=xi_mag)

        # The y-component should match interior exactly (same stencil)
        last_row = max(bdy_x.keys())
        np.testing.assert_allclose(
            result_bdy[last_row].C_y, result_int.C_y, atol=1e-3,
            err_msg="C_y should match interior (same y-direction stencil)",
        )


class Test2DBoundaryGroupVelocity:
    """2D boundary group velocity tests (36.2b)."""

    N_XI = 500

    def test_boundary_angle_distortion(self):
        """Boundary stencils distort group velocity angle relative to interior."""
        xi = np.linspace(0, np.pi, self.N_XI)
        theta = np.linspace(0.1, np.pi / 2 - 0.1, 200)
        xi_mag = 0.7

        for p, q, label in [(1, 1, "E2"), (2, 3, "E4")]:
            bdy_x = boundary_group_velocity(
                p=p, q=q, nextra=0, nu=1, sigma=0.1,
                kernel="tension", xi_array=xi,
            )
            int_y = interior_group_velocity(p=p, nu=1, xi_array=xi)

            result_bdy = boundary_group_velocity_2d(bdy_x, int_y, theta, xi_mag)
            result_int = anisotropy_profile(p=p, nu=1, theta_array=theta, xi_mag=xi_mag)

            # Every boundary row should have finite angle error
            for row_idx, r in result_bdy.items():
                assert np.all(np.isfinite(r.angle_error)), (
                    f"{label} row {row_idx}: non-finite angle error"
                )

            # Row 0 (closest to boundary) should have larger angle distortion
            # than the last row (furthest from boundary, closest to interior)
            row_0 = result_bdy[0]
            last_row = max(result_bdy.keys())
            row_last = result_bdy[last_row]

            distortion_0 = np.max(np.abs(row_0.angle_error - result_int.angle_error))
            distortion_last = np.max(np.abs(row_last.angle_error - result_int.angle_error))

            assert distortion_0 > distortion_last, (
                f"{label}: row 0 distortion ({distortion_0:.4f}) should exceed "
                f"last row distortion ({distortion_last:.4f})"
            )

            # Distortion at row 0 should be non-trivial (> 0.01 radians ~ 0.6 deg)
            assert distortion_0 > 0.01, (
                f"{label}: boundary distortion at row 0 too small ({distortion_0:.4f} rad)"
            )

    def test_corner_region(self):
        """At a corner, both directions use boundary stencils.

        Compute 2D group velocity by using boundary profiles for both x and y.
        The result should remain finite and bounded -- no blow-up from combining
        two boundary stencils.
        """
        xi = np.linspace(0, np.pi, self.N_XI)
        theta = np.linspace(0.1, np.pi / 2 - 0.1, 200)
        xi_mag = 0.5

        # E2 boundary stencils for both directions
        bdy = boundary_group_velocity(
            p=1, q=1, nextra=0, nu=1, sigma=0.1,
            kernel="tension", xi_array=xi,
        )

        # For a corner: use boundary profiles for BOTH x and y.
        # boundary_group_velocity_2d takes boundary x-profiles and an interior
        # y-profile. To simulate a corner, pass a boundary y-profile (row 0)
        # as the "interior_y" argument.
        corner_y = bdy[0]

        result_corner = boundary_group_velocity_2d(bdy, corner_y, theta, xi_mag)

        for row_idx, r in result_corner.items():
            # All values must be finite
            assert np.all(np.isfinite(r.speed)), (
                f"Corner row {row_idx}: non-finite speed"
            )
            assert np.all(np.isfinite(r.angle)), (
                f"Corner row {row_idx}: non-finite angle"
            )
            # Speed should not blow up (bounded by a generous factor)
            assert np.all(r.speed < 10), (
                f"Corner row {row_idx}: speed blow-up, max={np.max(r.speed):.2f}"
            )

        # Compare corner vs boundary-only-in-x: corner should be slower
        # because the y-direction also suffers boundary degradation
        int_y = interior_group_velocity(p=1, nu=1, xi_array=xi)
        result_bdy = boundary_group_velocity_2d(bdy, int_y, theta, xi_mag)

        row_0_corner_speed = np.mean(result_corner[0].speed)
        row_0_bdy_speed = np.mean(result_bdy[0].speed)

        # Corner degrades in both directions, so typically speed differs
        # (it could be either direction depending on stencil, but the results
        # should at least be finite and comparable in magnitude)
        assert abs(row_0_corner_speed - row_0_bdy_speed) < 5, (
            f"Corner vs boundary speed difference unreasonable: "
            f"corner={row_0_corner_speed:.3f}, bdy={row_0_bdy_speed:.3f}"
        )

    def test_no_outgoing_2d(self):
        """No 2D boundary modes have group velocity pointing into the domain
        at wavenumbers where the interior mode doesn't.

        At a left boundary (outward normal n = (-1, 0)), "into the domain"
        means C_x > 0. We check that wherever the interior C_x is <= 0
        (i.e., the interior mode does NOT propagate rightward), the boundary
        C_x is also not anomalously positive.
        """
        xi = np.linspace(0, np.pi, self.N_XI)
        # Use theta near 0 (wave mostly in x) to focus on x-component
        theta = np.linspace(0.05, np.pi / 4, 200)
        xi_mag = 0.7

        for p, q, label in [(1, 1, "E2"), (2, 3, "E4")]:
            bdy_x = boundary_group_velocity(
                p=p, q=q, nextra=0, nu=1, sigma=0.1,
                kernel="tension", xi_array=xi,
            )
            int_y = interior_group_velocity(p=p, nu=1, xi_array=xi)

            result_bdy = boundary_group_velocity_2d(bdy_x, int_y, theta, xi_mag)
            result_int = anisotropy_profile(
                p=p, nu=1, theta_array=theta, xi_mag=xi_mag,
            )

            # Interior C_x: where it's non-positive, boundary should not be
            # strongly positive (allow small tolerance for interpolation noise)
            int_nonpositive = result_int.C_x <= 0

            for row_idx, r in result_bdy.items():
                if not np.any(int_nonpositive):
                    continue
                # Where interior is non-positive, boundary C_x should not
                # exceed a small tolerance
                bdy_cx_at_nonpositive = r.C_x[int_nonpositive]
                max_outgoing = np.max(bdy_cx_at_nonpositive)
                assert max_outgoing < 0.1, (
                    f"{label} row {row_idx}: anomalous outgoing C_x = {max_outgoing:.4f} "
                    f"where interior C_x <= 0"
                )


class TestVaryingCoefficientGroupVelocity:
    """Tests for varying-coefficient group velocity analysis (36.3)."""

    N_XI = 200

    def test_varying_basic(self):
        """With constant a(x)=1, local GV matches interior GV everywhere."""
        from stencil_gen.interior import derive_interior, full_gamma_array

        p = 1  # E2
        coeffs = derive_interior(0, p, 1)
        w = [float(c) for c in full_gamma_array(coeffs)]
        nodes = list(range(-p, p + 1))

        xi_array = np.linspace(0, np.pi, self.N_XI)
        x = np.linspace(0, 1, 10)

        def weights_func(x_val):
            return w, nodes

        C = local_group_velocity(weights_func, x, xi_array)

        # Shape check
        assert C.shape == (len(x), len(xi_array))

        # All rows identical (constant coefficient)
        for i in range(1, len(x)):
            np.testing.assert_allclose(C[i], C[0], atol=1e-14)

        # Matches interior group velocity from exact formula
        C_interior = group_velocity_exact(w, 0, nodes, xi_array)
        np.testing.assert_allclose(C[0], C_interior, atol=1e-14)

    def test_constant_coefficient_uniform_gv(self):
        """With a(x) = 1 everywhere, local GV equals interior GV at all x."""
        from stencil_gen.interior import derive_interior, full_gamma_array

        for p in [1, 2, 3]:  # E2, E4, E6
            coeffs = derive_interior(0, p, 1)
            w = [float(c) for c in full_gamma_array(coeffs)]
            nodes = list(range(-p, p + 1))

            xi_array = np.linspace(0, np.pi, self.N_XI)
            x = np.linspace(0, 1, 20)

            def weights_func(x_val, _w=w, _n=nodes):
                return _w, _n

            C = local_group_velocity(weights_func, x, xi_array)
            C_int = group_velocity_exact(w, 0, nodes, xi_array)

            for i in range(len(x)):
                np.testing.assert_allclose(
                    C[i], C_int, atol=1e-14,
                    err_msg=f"E{2*p} at x={x[i]:.2f}",
                )

    def test_linear_coefficient_gv_variation(self):
        """With a(x) = 1 + 0.5*x, local GV varies smoothly with x."""
        from stencil_gen.interior import derive_interior, full_gamma_array

        p = 1  # E2
        coeffs = derive_interior(0, p, 1)
        w_base = np.array([float(c) for c in full_gamma_array(coeffs)])
        nodes = list(range(-p, p + 1))

        xi_array = np.linspace(0, np.pi, self.N_XI)
        x = np.linspace(0, 1, 30)

        def weights_func(x_val):
            # a(x) = 1 + 0.5*x scales the stencil weights
            a_x = 1.0 + 0.5 * x_val
            return (a_x * w_base).tolist(), nodes

        C = local_group_velocity(weights_func, x, xi_array)

        # C should scale linearly with a(x): C(x, xi) = a(x) * g(xi)
        # where g(xi) is the interior group velocity for unit coefficient
        g = group_velocity_exact(w_base.tolist(), 0, nodes, xi_array)
        for i in range(len(x)):
            a_x = 1.0 + 0.5 * x[i]
            np.testing.assert_allclose(
                C[i], a_x * g, atol=1e-13,
                err_msg=f"x={x[i]:.3f}, a(x)={a_x:.3f}",
            )

        # Verify smooth variation: max difference between adjacent x-points
        # should be bounded by the coefficient variation * max|g|
        dC = np.diff(C, axis=0)
        dx = np.diff(x)
        # dC/dx ≈ 0.5 * g(xi) for a(x) = 1 + 0.5*x
        for i in range(len(dx)):
            np.testing.assert_allclose(
                dC[i] / dx[i], 0.5 * g, atol=1e-10,
            )

    def test_sign_change_interface(self):
        """With a(x) changing sign, verify GV reversal at the interface.

        When a(x) < 0, the wave propagates in the -x direction, so the group
        velocity should be negative.  Trefethen (1983, Theorem 5) shows that
        interfaces where a(x) changes sign always have outgoing modes --
        energy radiates away from the sign change in both directions.
        """
        from stencil_gen.interior import derive_interior, full_gamma_array

        p = 1  # E2
        coeffs = derive_interior(0, p, 1)
        w_base = np.array([float(c) for c in full_gamma_array(coeffs)])
        nodes = list(range(-p, p + 1))

        xi_array = np.linspace(0.01, np.pi - 0.01, self.N_XI)
        x = np.linspace(-1, 1, 40)

        def weights_func(x_val):
            # a(x) = x: changes sign at x = 0
            a_x = x_val
            return (a_x * w_base).tolist(), nodes

        C = local_group_velocity(weights_func, x, xi_array)

        # At low wavenumbers where g(xi) > 0 (resolved waves):
        # C(x, xi) = a(x) * g(xi), so sign(C) = sign(a(x))
        g = group_velocity_exact(w_base.tolist(), 0, nodes, xi_array)
        low_xi_mask = xi_array < 1.0  # well-resolved waves
        assert np.any(low_xi_mask)

        # For x > 0 (a > 0): C should be positive at low xi
        pos_x = x > 0.2
        assert np.all(C[pos_x][:, low_xi_mask] > 0), "C should be positive for a(x) > 0"

        # For x < 0 (a < 0): C should be negative at low xi
        neg_x = x < -0.2
        assert np.all(C[neg_x][:, low_xi_mask] < 0), "C should be negative for a(x) < 0"

        # At x ≈ 0 (interface): C ≈ 0 (energy stalls at the interface)
        interface = np.abs(x) < 0.05
        if np.any(interface):
            assert np.all(np.abs(C[interface][:, low_xi_mask]) < 0.1)

    def test_ray_trace_uniform(self):
        """In a uniform medium, rays are straight lines at constant xi.

        Verifies to numerical precision that the RK4 integrator produces
        exact straight-line trajectories when dC/dx = 0.
        """
        from stencil_gen.interior import derive_interior, full_gamma_array

        p = 2  # E4
        coeffs = derive_interior(0, p, 1)
        w = [float(c) for c in full_gamma_array(coeffs)]
        nodes = list(range(-p, p + 1))

        xi_array = np.linspace(0.01, np.pi, self.N_XI)
        x_grid = np.linspace(-2, 3, 80)

        def weights_func(x_val):
            return w, nodes

        C_field = local_group_velocity(weights_func, x_grid, xi_array)

        # Test at several initial wavenumbers
        for xi_0 in [0.5, 1.0, 2.0]:
            result = ray_trace_group_velocity(
                C_field, x_grid, xi_array, xi_0, x_0=0.0, t_final=1.0, dt=0.01,
            )

            # xi constant
            np.testing.assert_allclose(result.xi, xi_0, atol=1e-10,
                                       err_msg=f"xi_0={xi_0}")

            # x linear
            C_val = float(np.interp(xi_0, xi_array,
                                     group_velocity_exact(w, 0, nodes, xi_array)))
            np.testing.assert_allclose(result.x, C_val * result.t, atol=1e-6,
                                       err_msg=f"xi_0={xi_0}")

    def test_ray_trace_refraction(self):
        """In a medium with linearly varying a(x), rays bend (refraction).

        For a(x) = 1 + epsilon*x with small epsilon, the ray equations are:
          dx/dt = a(x) * g(xi)
          dxi/dt = -epsilon * g(xi)  (since dC/dx = epsilon * g(xi))

        At fixed xi (valid for short times), dxi/dt ≈ -epsilon * g(xi_0),
        so xi decreases linearly.  Verify against this analytical prediction.
        """
        from stencil_gen.interior import derive_interior, full_gamma_array

        p = 1  # E2
        coeffs = derive_interior(0, p, 1)
        w_base = np.array([float(c) for c in full_gamma_array(coeffs)])
        nodes = list(range(-p, p + 1))

        xi_array = np.linspace(0.01, np.pi, 300)
        eps = 0.3
        x_grid = np.linspace(-2, 4, 100)

        def weights_func(x_val):
            a_x = 1.0 + eps * x_val
            return (a_x * w_base).tolist(), nodes

        C_field = local_group_velocity(weights_func, x_grid, xi_array)

        xi_0 = 1.0
        t_final = 0.5
        dt = 0.005

        result = ray_trace_group_velocity(
            C_field, x_grid, xi_array, xi_0, x_0=0.0, t_final=t_final, dt=dt,
        )

        # Analytical prediction for short times:
        # dxi/dt ≈ -eps * g(xi_0) where g is the interior 1D GV
        g_xi0 = float(np.interp(xi_0, xi_array,
                                 group_velocity_exact(w_base.tolist(), 0, nodes, xi_array)))
        xi_analytical = xi_0 - eps * g_xi0 * result.t

        # Should agree to O(eps * t_final) accuracy
        np.testing.assert_allclose(
            result.xi, xi_analytical, atol=0.05,
            err_msg="Refraction: xi should decrease linearly for small eps*t",
        )

        # xi should be monotonically changing (refraction is one-way here)
        dxi = np.diff(result.xi)
        assert np.all(dxi < 0), "xi should decrease monotonically for eps > 0"


class TestLocal2DVarying:
    """Tests for local_group_velocity_2d_varying and max_local_gv_error_2d."""

    @staticmethod
    def _interior_stencil(p: int, nu: int = 1):
        """Return (weights, offsets) for the interior scheme with half-bandwidth p."""
        from stencil_gen.interior import derive_interior, full_gamma_array

        coeffs = derive_interior(0, p, nu)
        w = np.array([float(c) for c in full_gamma_array(coeffs)])
        offsets = np.arange(-p, p + 1, dtype=float)
        return (w, offsets)

    def test_constant_coefficient_reduces_to_interior(self):
        """With c_x == 1, c_y == 0, the result should match interior_group_velocity."""
        p, nu = 2, 1  # E4
        stencil = self._interior_stencil(p, nu)
        xi = np.linspace(0.01, np.pi, 100)

        Ny, Nx = 5, 7
        c_x = np.ones((Ny, Nx))
        c_y = np.zeros((Ny, Nx))

        result = local_group_velocity_2d_varying(stencil, stencil, c_x, c_y, xi)

        # Reference: 1D interior profile
        ref_profile = interior_group_velocity(p, nu, xi)

        # C_x_field should be C_x(xi) at every point (c_x == 1)
        for i in range(Ny):
            for j in range(Nx):
                np.testing.assert_allclose(
                    result["C_x_field"][i, j, :],
                    ref_profile.group_velocity,
                    atol=1e-14,
                )
        # gv_error_x_field should match gv_error (c_x == 1)
        for i in range(Ny):
            for j in range(Nx):
                np.testing.assert_allclose(
                    result["gv_error_x_field"][i, j, :],
                    ref_profile.gv_error,
                    atol=1e-14,
                )
        # C_y_field and gv_error_y_field should be zero (c_y == 0)
        np.testing.assert_allclose(result["C_y_field"], 0.0, atol=1e-14)
        np.testing.assert_allclose(result["gv_error_y_field"], 0.0, atol=1e-14)

    def test_radial_flow_field(self):
        """max_local_gv_error_2d on the Brady-Livescu field is finite and reasonable."""
        from stencil_gen.benchmarks.brady_livescu_2d import make_coefficient_field

        p, nu = 2, 1  # E4
        stencil = self._interior_stencil(p, nu)
        xi = np.linspace(0.01, np.pi, 100)

        _, _, c_x, c_y = make_coefficient_field(31)
        result = local_group_velocity_2d_varying(stencil, stencil, c_x, c_y, xi)

        max_err = max_local_gv_error_2d(result)
        assert np.isfinite(max_err), "max_local_gv_error_2d should be finite"
        assert max_err > 0.0, "max_local_gv_error_2d should be positive"
        # E4 interior has some dispersion error, and c_x, c_y are O(1).
        # Near xi=pi the GV error is large (negative C), so the max over all
        # wavenumbers can exceed 2. Bound is a loose sanity check.
        assert max_err < 5.0, f"max_local_gv_error_2d unexpectedly large: {max_err}"

    def test_scalar_reduction_finite_for_both_schemes(self):
        """E2 and E4 both produce finite positive max_local_gv_error_2d on BL field."""
        from stencil_gen.benchmarks.brady_livescu_2d import make_coefficient_field

        xi = np.linspace(0.01, np.pi, 100)
        _, _, c_x, c_y = make_coefficient_field(31)

        for p in (1, 2):  # E2 (p=1), E4 (p=2)
            stencil = self._interior_stencil(p, nu=1)
            result = local_group_velocity_2d_varying(stencil, stencil, c_x, c_y, xi)
            max_err = max_local_gv_error_2d(result)
            assert np.isfinite(max_err), f"p={p}: max error should be finite"
            assert max_err > 0.0, f"p={p}: max error should be positive"


class TestAnisotropyOverField:
    """Tests for anisotropy_over_coefficient_field (41.7a)."""

    def test_constant_coefficient_uniform(self):
        """Uniform (c_x, c_y) = (1, 0) gives error matching anisotropy at theta=0."""
        N = 11
        c_x = np.ones((N, N))
        c_y = np.zeros((N, N))

        theta = np.linspace(0.01, np.pi / 2 - 0.01, 100)
        xi_mag = 1.0
        result = anisotropy_over_coefficient_field("E4", c_x, c_y, theta, xi_mag)

        # All local angles are 0.  The error at theta=0 should be the
        # anisotropy profile error at theta=0.
        ref = anisotropy_profile(2, 1, theta, xi_mag)
        err_at_0 = np.sqrt(
            (ref.C_x[0] - np.cos(theta[0])) ** 2
            + (ref.C_y[0] - np.sin(theta[0])) ** 2
        )
        assert result["max_aligned_error"] == pytest.approx(err_at_0, abs=1e-10)

    def test_radial_flow_field(self):
        """BL radial flow field produces finite positive max_aligned_error."""
        from stencil_gen.benchmarks.brady_livescu_2d import make_coefficient_field

        _, _, c_x, c_y = make_coefficient_field(31)
        theta = np.linspace(0.01, np.pi / 2 - 0.01, 200)
        xi_mag = 1.0
        result = anisotropy_over_coefficient_field("E4", c_x, c_y, theta, xi_mag)

        assert np.isfinite(result["max_aligned_error"])
        assert result["max_aligned_error"] > 0.0
        # E4 at xi_mag=1 has small but nonzero anisotropy
        assert result["max_aligned_error"] < 1.0
        # worst_point should be a valid index
        i, j = result["worst_point"]
        assert 0 <= i < 31
        assert 0 <= j < 31

    def test_higher_xi_mag_increases_error(self):
        """Anisotropy error grows with wavenumber magnitude."""
        from stencil_gen.benchmarks.brady_livescu_2d import make_coefficient_field

        _, _, c_x, c_y = make_coefficient_field(21)
        theta = np.linspace(0.01, np.pi / 2 - 0.01, 100)

        err_low = anisotropy_over_coefficient_field(
            "E4", c_x, c_y, theta, xi_mag=0.5
        )["max_aligned_error"]
        err_high = anisotropy_over_coefficient_field(
            "E4", c_x, c_y, theta, xi_mag=2.0
        )["max_aligned_error"]
        assert err_high > err_low

    def test_e2_larger_error_than_e4(self):
        """E2 has worse anisotropy than E4 at a given xi_mag."""
        N = 21
        c_x = np.ones((N, N)) * 0.7
        c_y = np.ones((N, N)) * 0.7
        theta = np.linspace(0.01, np.pi / 2 - 0.01, 100)
        xi_mag = 1.5

        err_e2 = anisotropy_over_coefficient_field(
            "E2", c_x, c_y, theta, xi_mag
        )["max_aligned_error"]
        err_e4 = anisotropy_over_coefficient_field(
            "E4", c_x, c_y, theta, xi_mag
        )["max_aligned_error"]
        assert err_e2 > err_e4
