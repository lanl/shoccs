"""Unit tests for sweeps.gv_objectives helper wrappers.

These tests exercise the thin scalar helpers used by sweep scripts.  They
should all run well under 2 seconds in aggregate so they stay in the default
(non-slow) suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from stencil_gen.phs import build_diff_matrix_rbf

from sweeps import _common as sweeps_common
from sweeps import (
    brady2d_sweep,
    epsilon_sweep,
    footprint_sweep,
    tension_penalty_sweep,
    tension_sweep,
)
from sweeps.gv_objectives import (
    boundary_gv_error_max,
    cutcell_gv_min_C,
    gv_score_from_matrix,
    interior_cutoff_fraction,
    interior_gv_error_max,
)

KNOWN_VALUES_PATH = Path(__file__).parent.parent / "sweeps" / "known_values.json"


def _load_known():
    with open(KNOWN_VALUES_PATH) as f:
        return json.load(f)


def test_interior_gv_error_max_e2_positive_finite():
    val = interior_gv_error_max(p=1, nu=1, n_xi=100)
    assert np.isfinite(val)
    assert val > 0.0


def test_interior_cutoff_fraction_improves_with_order():
    f2 = interior_cutoff_fraction(p=1, nu=1, n_xi=200)
    f4 = interior_cutoff_fraction(p=2, nu=1, n_xi=200)
    assert f4 > f2  # higher-order scheme resolves more of the spectrum


def test_interior_cutoff_fraction_in_unit_interval():
    frac = interior_cutoff_fraction(p=2, nu=1, n_xi=100)
    assert 0.0 < frac <= 1.0 + 1e-12


def test_boundary_gv_error_max_e2_tension_at_known_sigma():
    kv = _load_known()
    sigma = kv["E2_1"]["tension"]["sigma"]
    val = boundary_gv_error_max(
        p=1, q=1, nextra=1, nu=1, sigma=sigma, kernel="tension", n_xi=100,
    )
    assert np.isfinite(val)
    assert val > 0.0


def test_boundary_gv_error_max_e4_tension_at_known_sigma():
    kv = _load_known()
    sigma = kv["E4_1"]["tension"]["sigma"]
    val = boundary_gv_error_max(
        p=2, q=3, nextra=0, nu=1, sigma=sigma, kernel="tension", n_xi=100,
    )
    assert np.isfinite(val)
    assert val > 0.0


def test_cutcell_gv_min_C_e2_returns_finite_tuple():
    """cutcell_gv_min_C on E2_1 returns (finite float, bool) at small psi grid."""
    from stencil_gen.temo import E2_1

    result = cutcell_gv_min_C(
        E2_1,
        psi_values=np.linspace(0.05, 0.95, 5),
        alpha_values={},
        n_xi=100,
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
    min_C, has_sign_reversal = result
    assert np.isfinite(min_C)
    assert isinstance(has_sign_reversal, bool)


def test_gv_score_from_matrix_matches_boundary_helper():
    """gv_score_from_matrix on a real D should agree with boundary_gv_error_max."""
    kv = _load_known()
    sigma = kv["E2_1"]["tension"]["sigma"]
    D = build_diff_matrix_rbf(
        40, p=1, q=1, epsilon=sigma, kernel="tension", nu=1, nextra=1,
    )
    from_matrix = gv_score_from_matrix(D, n_xi=100)
    from_helper = boundary_gv_error_max(
        p=1, q=1, nextra=1, nu=1, sigma=sigma, kernel="tension", n_xi=100,
    )
    assert np.isclose(from_matrix["max_gv_error"], from_helper, rtol=1e-12)
    assert 0.0 < from_matrix["min_cutoff_xi"] <= np.pi + 1e-12


def test_gv_score_from_matrix_small_hardcoded():
    """Deterministic smoke: tiny hand-built matrix produces finite results."""
    # 5-point matrix: first two rows boundary-like (start at column 0),
    # remaining rows interior-shifted so they are ignored by the scanner.
    D = np.zeros((5, 5))
    # Row 0: forward difference from column 0
    D[0, 0] = -1.0
    D[0, 1] = 1.0
    # Row 1: still starts at column 0 (boundary block)
    D[1, 0] = -0.5
    D[1, 1] = 0.0
    D[1, 2] = 0.5
    # Rows 2..4: centered interior (leftmost nonzero > 0 → scanner stops)
    for i in range(2, 4):
        D[i, i - 1] = -0.5
        D[i, i + 1] = 0.5
    score = gv_score_from_matrix(D, n_xi=50)
    assert np.isfinite(score["max_gv_error"])
    assert np.isfinite(score["min_cutoff_xi"])
    assert score["max_gv_error"] > 0.0


def _seed_kv_with_keys(
    path: Path,
    scheme_key: str,
    primary_key: str,
    secondary_key: str,
    primary_extras: dict,
    secondary_extras: dict,
    *,
    include_params: bool = True,
) -> dict:
    """Seed a temp known_values.json with sentinel-bearing primary/secondary entries.

    Used by the per-sweep merge regression tests.  Pre-existing keys that the
    sweep's ``main()`` is NOT allowed to clobber are placed on both entries:
    ``preexisting_extra_key`` on the primary entry and ``preexisting_gv_extra``
    on the secondary entry.  A stable-at list and ``gv_error=1.234`` sentinel
    are also seeded so tests can assert refresh vs preservation semantics.
    When ``include_params=False`` (footprint), the ``params`` sub-entry is
    omitted because footprint entries live directly under ``kv["footprint"]``.
    """
    scheme_entry: dict = {}
    if include_params:
        scheme_entry["params"] = {"p": 1, "q": 1, "nextra": 1, "nu": 1}
    scheme_entry[primary_key] = {
        **primary_extras,
        "stable_at": [20, 40, 80],
        "gv_error": 1.234,
        "preexisting_extra_key": "survive",
    }
    scheme_entry[secondary_key] = {
        **secondary_extras,
        "gv_error": 1.234,
        "stable_at": [20, 40],
        "preexisting_gv_extra": "survive_gv",
    }
    seed = {scheme_key: scheme_entry}
    with open(path, "w") as f:
        json.dump(seed, f, indent=2)
    return seed


def test_tension_sweep_main_merges_known_values(tmp_path, monkeypatch, capsys):
    """Regression for 40.2d: --update-known-values must merge, not overwrite.

    A non-GV --update-known-values invocation must preserve any pre-existing
    keys that an earlier --include-gv run wrote (gv_error on tension, the
    full tension_gv entry).  A subsequent --include-gv invocation must
    refresh those keys without dropping unrelated keys (preexisting_extra_key).
    """
    kv_path = tmp_path / "known_values.json"
    monkeypatch.setattr(sweeps_common, "KNOWN_VALUES_PATH", kv_path)
    _seed_kv_with_keys(
        kv_path,
        scheme_key="E2_1",
        primary_key="tension",
        secondary_key="tension_gv",
        primary_extras={"sigma": 6.0},
        secondary_extras={"sigma": 5.5},
    )

    rc = tension_sweep.main([
        "--scheme", "E2",
        "--n-sigma", "5",
        "--n-values", "20",
        "--update-known-values",
    ])
    assert rc == 0
    capsys.readouterr()

    with open(kv_path) as f:
        after_non_gv = json.load(f)
    tension = after_non_gv["E2_1"]["tension"]
    assert tension["gv_error"] == 1.234
    assert tension["preexisting_extra_key"] == "survive"
    assert "sigma" in tension and "stable_at" in tension
    assert after_non_gv["E2_1"]["tension_gv"] == {
        "sigma": 5.5,
        "gv_error": 1.234,
        "stable_at": [20, 40],
        "preexisting_gv_extra": "survive_gv",
    }

    rc = tension_sweep.main([
        "--scheme", "E2",
        "--n-sigma", "5",
        "--n-values", "20",
        "--include-gv",
        "--update-known-values",
    ])
    assert rc == 0
    capsys.readouterr()

    with open(kv_path) as f:
        after_gv = json.load(f)
    tension = after_gv["E2_1"]["tension"]
    assert tension["preexisting_extra_key"] == "survive"
    assert "sigma" in tension
    assert "stable_at" in tension
    assert np.isfinite(tension["gv_error"])
    tension_gv = after_gv["E2_1"]["tension_gv"]
    assert {"sigma", "gv_error", "stable_at"} <= set(tension_gv)
    assert tension_gv["preexisting_gv_extra"] == "survive_gv"
    assert np.isfinite(tension_gv["sigma"])
    assert np.isfinite(tension_gv["gv_error"])
    assert isinstance(tension_gv["stable_at"], list)


def test_epsilon_sweep_main_merges_known_values(tmp_path, monkeypatch, capsys):
    """Regression for 40.3c: epsilon_sweep --update-known-values must merge.

    Mirrors test_tension_sweep_main_merges_known_values.  A non-GV
    --update-known-values invocation must preserve any pre-existing keys
    that an earlier --include-gv run wrote on both ``{kernel}`` and
    ``{kernel}_gv`` entries.  A subsequent --include-gv invocation must
    refresh the keys this run owns without dropping unrelated keys
    (preexisting_extra_key on ``{kernel}``, preexisting_gv_extra on
    ``{kernel}_gv``).
    """
    kernel = "gaussian"
    gv_key = f"{kernel}_gv"
    kv_path = tmp_path / "known_values.json"
    monkeypatch.setattr(sweeps_common, "KNOWN_VALUES_PATH", kv_path)
    _seed_kv_with_keys(
        kv_path,
        scheme_key="E2_1",
        primary_key=kernel,
        secondary_key=gv_key,
        primary_extras={"epsilon": 1.0},
        secondary_extras={"epsilon": 0.9},
    )

    rc = epsilon_sweep.main([
        "--scheme", "E2",
        "--kernel", kernel,
        "--n-eps", "5",
        "--n-values", "20",
        "--update-known-values",
    ])
    assert rc == 0
    capsys.readouterr()

    with open(kv_path) as f:
        after_non_gv = json.load(f)
    kernel_entry = after_non_gv["E2_1"][kernel]
    assert kernel_entry["gv_error"] == 1.234
    assert kernel_entry["preexisting_extra_key"] == "survive"
    assert "epsilon" in kernel_entry and "stable_at" in kernel_entry
    assert after_non_gv["E2_1"][gv_key] == {
        "epsilon": 0.9,
        "gv_error": 1.234,
        "stable_at": [20, 40],
        "preexisting_gv_extra": "survive_gv",
    }

    rc = epsilon_sweep.main([
        "--scheme", "E2",
        "--kernel", kernel,
        "--n-eps", "5",
        "--n-values", "20",
        "--include-gv",
        "--update-known-values",
    ])
    assert rc == 0
    capsys.readouterr()

    with open(kv_path) as f:
        after_gv = json.load(f)
    kernel_entry = after_gv["E2_1"][kernel]
    assert kernel_entry["preexisting_extra_key"] == "survive"
    assert "epsilon" in kernel_entry
    assert "stable_at" in kernel_entry
    assert np.isfinite(kernel_entry["gv_error"])
    gv_entry = after_gv["E2_1"][gv_key]
    assert {"epsilon", "gv_error", "stable_at"} <= set(gv_entry)
    assert gv_entry["preexisting_gv_extra"] == "survive_gv"
    assert np.isfinite(gv_entry["epsilon"])
    assert np.isfinite(gv_entry["gv_error"])
    assert isinstance(gv_entry["stable_at"], list)


def test_tension_penalty_sweep_main_merges_known_values(tmp_path, monkeypatch, capsys):
    """Regression for 40.4c/40.4d: tension_penalty --update-known-values must merge.

    Mirrors test_tension_sweep_main_merges_known_values and
    test_epsilon_sweep_main_merges_known_values.  tension_penalty has no
    --include-gv flag (GV is always computed by eval_point), so only one
    invocation path exists.  The merge must preserve preexisting keys on
    both ``tension_penalty`` and ``tension_penalty_gv`` entries while
    refreshing the keys this invocation owns.  After 40.4d, the GV entry
    is expected to gain a non-empty ``stable_at`` list from the cross-grid
    re-check.
    """
    kv_path = tmp_path / "known_values.json"
    monkeypatch.setattr(sweeps_common, "KNOWN_VALUES_PATH", kv_path)
    _seed_kv_with_keys(
        kv_path,
        scheme_key="E2_1",
        primary_key="tension_penalty",
        secondary_key="tension_penalty_gv",
        primary_extras={"sigma": 6.0, "gamma": 0.0},
        secondary_extras={"sigma": 5.5, "gamma": 0.1},
    )

    rc = tension_penalty_sweep.main([
        "--scheme", "E2",
        "--n-sigma", "5",
        "--n-gamma", "5",
        "--update-known-values",
    ])
    assert rc == 0
    capsys.readouterr()

    with open(kv_path) as f:
        after = json.load(f)
    tp = after["E2_1"]["tension_penalty"]
    assert tp["preexisting_extra_key"] == "survive"
    assert "sigma" in tp and "gamma" in tp and "stable_at" in tp
    assert np.isfinite(tp["gv_error"])
    # gv_error must have been refreshed away from the seeded sentinel.
    assert tp["gv_error"] != 1.234

    tp_gv = after["E2_1"]["tension_penalty_gv"]
    assert tp_gv["preexisting_gv_extra"] == "survive_gv"
    assert {"sigma", "gamma", "gv_error", "stable_at"} <= set(tp_gv)
    assert np.isfinite(tp_gv["sigma"])
    assert np.isfinite(tp_gv["gamma"])
    assert np.isfinite(tp_gv["gv_error"])
    assert isinstance(tp_gv["stable_at"], list)
    assert len(tp_gv["stable_at"]) > 0
    assert set(tp_gv["stable_at"]) <= {20, 40, 80, 160}


def test_footprint_sweep_main_merges_known_values(tmp_path, monkeypatch, capsys):
    """Regression for 40.5c/40.5d: footprint --update-known-values must merge.

    Mirrors the tension / epsilon / tension-penalty merge regression tests, but
    pins the two-level merge pattern unique to footprint: entries live directly
    under ``kv["footprint"]`` (no ``scheme_key`` wrapper), and the merge loop
    must preserve both (a) a pre-existing primary entry keyed by a sigma value
    the current run does NOT reproduce (e.g. ``E4_nextra0_tension_3`` — the
    real-world key at n-sigma 5 best_sigma != 3.0), and (b) a pre-existing
    ``_tension_gv`` entry with a sentinel.  Also pins the 40.5d filter: at
    ``--n-sigma 5`` the GV optimum for nextra=1 collapses to sigma=0, which
    must NOT be persisted as a tension entry (it would mis-label a PHS
    baseline point as tension).
    """
    kv_path = tmp_path / "known_values.json"
    monkeypatch.setattr(sweeps_common, "KNOWN_VALUES_PATH", kv_path)
    _seed_kv_with_keys(
        kv_path,
        scheme_key="footprint",
        primary_key="E4_nextra0_tension_3",
        secondary_key="E4_nextra0_tension_gv",
        primary_extras={"nextra": 0, "sigma": 3.0},
        secondary_extras={"nextra": 0, "sigma": 5.5},
        include_params=False,
    )

    # First run: no --include-gv.  The smoke run's best_sigma for nextra=0 is
    # 0.0 (PHS baseline, filtered out of tension entries), so the run does not
    # touch either seeded key.  Both sentinels must survive.
    rc = footprint_sweep.main([
        "--n-sigma", "5",
        "--update-known-values",
    ])
    assert rc == 0
    capsys.readouterr()

    with open(kv_path) as f:
        after_non_gv = json.load(f)
    footprint = after_non_gv["footprint"]
    assert footprint["E4_nextra0_tension_3"]["preexisting_extra_key"] == "survive"
    assert footprint["E4_nextra0_tension_3"]["gv_error"] == 1.234
    # The non-GV path does not write _tension_gv keys, so the seeded entry
    # must be byte-identical to the seed.
    assert footprint["E4_nextra0_tension_gv"] == {
        "nextra": 0,
        "sigma": 5.5,
        "gv_error": 1.234,
        "stable_at": [20, 40],
        "preexisting_gv_extra": "survive_gv",
    }

    # Second run: --include-gv.  The primary sentinel must still survive
    # (best_sigma != 3.0 for nextra=0 at --n-sigma 5, so the run still does not
    # touch E4_nextra0_tension_3).  The secondary sentinel must survive because
    # the per-entry merge is additive.  gv_error is refreshed.
    rc = footprint_sweep.main([
        "--n-sigma", "5",
        "--include-gv",
        "--update-known-values",
    ])
    assert rc == 0
    capsys.readouterr()

    with open(kv_path) as f:
        after_gv = json.load(f)
    footprint = after_gv["footprint"]

    # Primary entry: seed was keyed at sigma=3, run does not reproduce it, so
    # the whole entry survives untouched including both sentinel and gv_error.
    t3 = footprint["E4_nextra0_tension_3"]
    assert t3["preexisting_extra_key"] == "survive"
    assert t3["gv_error"] == 1.234
    assert t3["sigma"] == 3.0

    # Secondary entry: the run writes a fresh E4_nextra0_tension_gv (nextra=0
    # GV-optimal sigma at --n-sigma 5 is 50.0), so the merge must overlay
    # the new {nextra, sigma, gv_error, stable_at} keys on top of the seeded
    # dict without dropping the preexisting_gv_extra sentinel.
    gv0 = footprint["E4_nextra0_tension_gv"]
    assert gv0["preexisting_gv_extra"] == "survive_gv"
    assert {"nextra", "sigma", "gv_error", "stable_at"} <= set(gv0)
    assert gv0["nextra"] == 0
    assert np.isfinite(gv0["sigma"])
    assert gv0["sigma"] > 0  # 40.5d filter: must be a real tension sigma
    assert np.isfinite(gv0["gv_error"])
    assert gv0["gv_error"] != 1.234  # refreshed
    assert isinstance(gv0["stable_at"], list)
    assert len(gv0["stable_at"]) > 0
    assert set(gv0["stable_at"]) <= {20, 40, 80, 160}

    # 40.5d filter: nextra=1's GV optimum at --n-sigma 5 is sigma=0 (PHS
    # baseline), which must be filtered before being persisted as a
    # _tension_gv entry.  The seed did not include this key, so it must
    # remain absent after both runs.
    assert "E4_nextra1_tension_gv" not in footprint


def test_footprint_primary_gv_error_bit_exact_at_persisted_sigma(
    tmp_path, monkeypatch, capsys,
):
    """40.8f: footprint primary entries must persist gv_error computed at the
    *rounded* sigma so that ``(sigma, gv_error)`` is bit-exact self-consistent.

    The existing ``test_footprint_primary_tension_gv_error_match`` regression in
    test_phs.py uses a 0.1% strict tolerance, but at smoke-run resolution every
    populated nextra has a ``best_sigma`` whose 4dp rounding drift is ~1e-6
    relative — well below 0.1% — so a regression that re-introduces
    ``sigma=float(res["best_sigma"])`` instead of ``sigma=best_sigma_rounded``
    would not trip the strict-tolerance gate.  This test is the mechanical
    fallback (b) the plan calls out: it asserts bit-exact equality between the
    persisted ``gv_error`` and ``boundary_gv_error_max`` re-evaluated at the
    persisted ``sigma``.  ``boundary_gv_error_max`` is a deterministic numpy
    pipeline, so any non-zero parameter drift produces a non-zero gv_error
    delta — converting the smoke-run's 1e-6 relative drift into a non-bit-equal
    failure that the test can detect, regardless of how forgiving the strict
    tolerance is.

    Mutation check (run by hand during 40.8f review): reverting line 460-463 of
    ``footprint_sweep.py`` to ``sigma=float(res["best_sigma"])`` causes this
    test to fail with::

        AssertionError: E4_nextra1_tension_1: persisted gv_error
            4.9596397695... != 4.9596374230... rebuilt at sigma=1.3852

    confirming that the bit-exact gate trips on the exact bug 40.8d eliminated.
    """
    kv_path = tmp_path / "known_values.json"
    monkeypatch.setattr(sweeps_common, "KNOWN_VALUES_PATH", kv_path)

    rc = footprint_sweep.main([
        "--n-sigma", "5",
        "--include-gv",
        "--update-known-values",
    ])
    assert rc == 0
    capsys.readouterr()

    with open(kv_path) as f:
        kv = json.load(f)
    footprint = kv["footprint"]

    primary_entries = [
        (key, entry) for key, entry in footprint.items()
        if "_tension_" in key
        and not key.endswith("_gv")
        and "gv_error" in entry
    ]
    assert primary_entries, (
        "smoke-run footprint sweep produced no primary _tension_N entries with "
        "gv_error — the 40.8f gate has nothing to validate; check that "
        "footprint_sweep populates at least one stable nextra at --n-sigma 5"
    )

    for key, entry in primary_entries:
        rebuilt = float(boundary_gv_error_max(
            p=footprint_sweep.P,
            q=footprint_sweep.Q,
            nextra=entry["nextra"],
            nu=footprint_sweep.NU,
            sigma=entry["sigma"],
            kernel="tension",
        ))
        assert entry["gv_error"] == rebuilt, (
            f"{key}: persisted gv_error {entry['gv_error']!r} != {rebuilt!r} "
            f"rebuilt at sigma={entry['sigma']} — the (sigma, gv_error) pair "
            f"is not bit-exact self-consistent, indicating gv_error was "
            f"computed at an un-rounded sigma (40.8d regression)"
        )


def test_tension_penalty_primary_gv_error_bit_exact_at_persisted_sigma(
    tmp_path, monkeypatch, capsys,
):
    """40.8g: tension_penalty primary + secondary entries must persist gv_error
    computed at the *rounded* (sigma, gamma) so that the persisted triple is
    bit-exact self-consistent.

    Mechanical gate mirroring 40.8f's footprint bit-exact test.  Runs
    ``tension_penalty_sweep.main`` against a ``tmp_path``-redirected
    ``KNOWN_VALUES_PATH``, then for both the primary ``tension_penalty`` entry
    and the secondary ``tension_penalty_gv`` entry re-evaluates
    ``build_diff_matrix_rbf_penalty`` + ``gv_score_from_matrix`` at the
    persisted ``(sigma, gamma)`` and asserts **bit-exact** equality (``==``,
    not tolerance) with the persisted ``gv_error``.
    """
    from stencil_gen.phs import build_diff_matrix_rbf_penalty

    kv_path = tmp_path / "known_values.json"
    monkeypatch.setattr(sweeps_common, "KNOWN_VALUES_PATH", kv_path)

    rc = tension_penalty_sweep.main([
        "--scheme", "E2",
        "--n-sigma", "5",
        "--n-gamma", "5",
        "--update-known-values",
    ])
    assert rc == 0
    capsys.readouterr()

    with open(kv_path) as f:
        kv = json.load(f)
    e2 = kv["E2_1"]
    params = {"p": 1, "q": 1, "nextra": 1, "nu": 1}

    checked = 0
    for key in ("tension_penalty", "tension_penalty_gv"):
        entry = e2.get(key)
        if not isinstance(entry, dict) or "gv_error" not in entry:
            continue
        sigma = entry["sigma"]
        gamma = entry["gamma"]
        stored = entry["gv_error"]
        D = build_diff_matrix_rbf_penalty(
            40, params["p"], params["q"], sigma, "tension",
            params["nu"], params["nextra"], gamma=gamma,
        )
        rebuilt = float(gv_score_from_matrix(D)["max_gv_error"])
        assert stored == rebuilt, (
            f"E2_1.{key}: persisted gv_error {stored!r} != {rebuilt!r} "
            f"rebuilt at sigma={sigma}, gamma={gamma} — the (sigma, gamma, "
            f"gv_error) triple is not bit-exact self-consistent, indicating "
            f"gv_error was computed at un-rounded params (40.8g regression)"
        )
        checked += 1
    assert checked >= 1, (
        "tension_penalty_sweep smoke run produced no E2_1.tension_penalty "
        "entry with gv_error — the 40.8g bit-exact gate has nothing to "
        "validate"
    )


def test_check_gks_advisory_tension_e2_no_false_positives(capsys):
    """40.7b: --check-gks on a known-stable E2 tension case must not warn.

    The 40.7a verification confirmed that ``tension --scheme E2 --n-sigma 5``
    produces zero outgoing boundary modes at the discovered sigma*.  This
    test pins that as a regression: the sweep must exit 0, the GKS advisory
    block must appear, and no ``WARNING:`` line (no false-positive outgoing
    mode) may be printed.
    """
    rc = tension_sweep.main([
        "--scheme", "E2",
        "--n-sigma", "5",
        "--n-values", "20",
        "--check-gks",
    ])
    assert rc == 0
    captured = capsys.readouterr()
    out = captured.out
    assert "GKS advisory" in out
    assert "no outgoing boundary modes detected" in out
    assert "WARNING:" not in out


def test_check_gks_advisory_epsilon_e2_gaussian_no_false_positives(capsys):
    """40.7c: --check-gks on a known-stable E2 gaussian case must not warn.

    Mirrors ``test_check_gks_advisory_tension_e2_no_false_positives`` for the
    epsilon sweep so a future refactor of ``epsilon_sweep.main``'s
    ``--check-gks`` argparse entry, the passthrough in ``sweeps/__main__.py``,
    or the ``print_gks_advisory`` call site inside ``run_epsilon_sweep`` cannot
    silently break the wiring.  The 40.7a verification confirmed that gaussian
    at the eps -> 0 polynomial-reproduction limit has 1 inspected boundary mode
    and zero outgoing boundary modes.  Plan 46.3b.1.2 raised the default
    ``--eps-floor`` to 1.5, which moves the smoke-run optimum into the upper
    stable basin where an outgoing mode does appear; pass ``--eps-floor 0.0``
    here to recover the eps -> 0 case that 40.7a was pinned to.
    """
    rc = epsilon_sweep.main([
        "--scheme", "E2",
        "--kernel", "gaussian",
        "--n-eps", "5",
        "--n-values", "40",
        "--eps-floor", "0.0",
        "--check-gks",
    ])
    assert rc == 0
    captured = capsys.readouterr()
    out = captured.out
    assert "GKS advisory" in out
    assert "kernel=gaussian" in out
    assert "no outgoing boundary modes detected" in out
    assert "WARNING:" not in out


def test_check_gks_advisory_helper_clean_matrix(capsys):
    """40.7b smoke: ``print_gks_advisory`` on a clean diff matrix returns 0.

    Builds the same RBF differentiation matrix that the 40.7a verification
    confirmed has zero outgoing boundary modes (E2 tension at sigma*=0.0,
    n=20) and asserts the helper returns 0 and prints the clean line.
    """
    from sweeps.gv_objectives import print_gks_advisory

    D = build_diff_matrix_rbf(
        20, p=1, q=1, epsilon=0.0, kernel="tension", nu=1, nextra=1,
    )
    n_outgoing = print_gks_advisory(D, label="smoke")
    assert n_outgoing == 0
    captured = capsys.readouterr()
    assert "GKS advisory (smoke)" in captured.out
    assert "no outgoing boundary modes detected" in captured.out
    assert "WARNING:" not in captured.out


# ---------------------------------------------------------------------------
# 42.8a-fu1: unit tests for sweeps/brady2d_sweep.py
# ---------------------------------------------------------------------------


def _make_stub_report(
    *,
    verdict: str = "pass",
    failed_layer: int | None = None,
    l1_err: float = 1e-3,
    l3_eig: float = -1e-4,
    l6_tgb: float | None = None,
) -> "brady2d_sweep.StabilityReport":
    """Build a minimal StabilityReport for stubbing brady2d_stability_score."""
    from stencil_gen.brady2d_stability import StabilityReport

    report = StabilityReport.empty()
    report.layer1 = {"boundary_gv_err": float(l1_err)}
    report.layer3 = {"max_stab_eig": float(l3_eig)}
    if l6_tgb is not None:
        report.layer6 = {
            "spectral_abscissa": -1e-4,
            "transient_growth_bound": float(l6_tgb),
            "non_normality_report": None,
        }
    report.overall_verdict = verdict
    report.failed_layer = failed_layer
    return report


def _install_stub_score(monkeypatch, recorder: list[dict]):
    """Replace brady2d_stability_score with a recorder that returns pass reports."""

    def _stub(scheme, kernel, params, *, max_layer, **kwargs):
        recorder.append({
            "scheme": scheme,
            "kernel": kernel,
            "params": dict(params),
            "max_layer": max_layer,
            "kwargs": dict(kwargs),
        })
        return _make_stub_report(verdict="pass")

    monkeypatch.setattr(brady2d_sweep, "brady2d_stability_score", _stub)


def test_brady2d_sweep_main_classical_single_point(monkeypatch, capsys):
    """42.8a-fu1 (1): --kernel classical runs once with CLASSICAL_E4_ALPHA and no --param-range."""
    calls: list[dict] = []
    _install_stub_score(monkeypatch, calls)

    rc = brady2d_sweep.main([
        "--scheme", "E4",
        "--kernel", "classical",
        "--max-layer", "1",
    ])
    assert rc == 0
    assert len(calls) == 1, f"expected one sweep point for classical, got {len(calls)}"
    call = calls[0]
    assert call["kernel"] == "classical"
    assert call["scheme"] == "E4"
    assert call["params"] == {"alpha": list(brady2d_sweep.CLASSICAL_E4_ALPHA)}
    assert call["max_layer"] == 1

    out = capsys.readouterr().out
    assert "brady2d sweep" in out
    assert "pass" in out


def test_brady2d_sweep_main_spline_requires_param_range(monkeypatch, capsys):
    """42.8a-fu1 (2): spline kernels without --param-range exit 2 with a diagnostic."""
    # Even though the stub would accept any call, _build_param_values should
    # raise before brady2d_stability_score is ever invoked.
    calls: list[dict] = []
    _install_stub_score(monkeypatch, calls)

    rc = brady2d_sweep.main([
        "--scheme", "E4",
        "--kernel", "tension",
        "--max-layer", "1",
    ])
    assert rc == 2
    assert calls == [], "spline kernel must not invoke the stability score without --param-range"

    captured = capsys.readouterr()
    assert "--param-range" in captured.err
    assert "tension" in captured.err


@pytest.mark.parametrize(
    "kernel,expected_key",
    [
        ("tension", "sigma"),
        ("gaussian", "epsilon"),
        ("multiquadric", "epsilon"),
    ],
)
def test_brady2d_sweep_main_param_range_parsing(
    monkeypatch, capsys, kernel, expected_key,
):
    """42.8a-fu1 (3): --param-range 2 4 3 produces 3 points at {2.0, 3.0, 4.0} with the right key."""
    calls: list[dict] = []
    _install_stub_score(monkeypatch, calls)

    rc = brady2d_sweep.main([
        "--scheme", "E4",
        "--kernel", kernel,
        "--param-range", "2", "4", "3",
        "--max-layer", "1",
    ])
    assert rc == 0
    assert len(calls) == 3, f"expected 3 sweep points from --param-range 2 4 3, got {len(calls)}"

    observed_values = [c["params"][expected_key] for c in calls]
    np.testing.assert_allclose(observed_values, [2.0, 3.0, 4.0], rtol=0, atol=0)

    # And the other scalar name must NOT appear in params.
    off_key = "epsilon" if expected_key == "sigma" else "sigma"
    for c in calls:
        assert off_key not in c["params"]
        assert c["kernel"] == kernel

    capsys.readouterr()  # consume


def test_brady2d_sweep_main_update_known_values(tmp_path, monkeypatch, capsys):
    """42.8a-fu1 (4): --update-known-values writes brady2d_sweep.<scheme>.<kernel> and preserves siblings."""
    kv_path = tmp_path / "known_values.json"
    # Seed an unrelated top-level key that must not be clobbered.
    kv_path.write_text(json.dumps({
        "E4_1": {"tension": {"sigma": 3.0, "preexisting_extra_key": "survive"}},
        "footprint": {"preexisting": "survive"},
    }))
    monkeypatch.setattr(sweeps_common, "KNOWN_VALUES_PATH", kv_path)

    calls: list[dict] = []
    _install_stub_score(monkeypatch, calls)

    rc = brady2d_sweep.main([
        "--scheme", "E4",
        "--kernel", "tension",
        "--param-range", "2", "4", "3",
        "--max-layer", "1",
        "--update-known-values",
    ])
    assert rc == 0
    capsys.readouterr()

    with open(kv_path) as f:
        after = json.load(f)

    # Unrelated siblings preserved.
    assert after["E4_1"]["tension"]["preexisting_extra_key"] == "survive"
    assert after["footprint"] == {"preexisting": "survive"}

    # brady2d_sweep entry present under the right scheme/kernel.
    assert "brady2d_sweep" in after
    bucket = after["brady2d_sweep"]["E4"]["tension"]
    assert bucket["scheme"] == "E4"
    assert bucket["kernel"] == "tension"
    assert bucket["param_name"] == "sigma"
    assert bucket["max_layer"] == 1
    assert len(bucket["points"]) == 3
    params_values = [pt["params_dict"]["sigma"] for pt in bucket["points"]]
    np.testing.assert_allclose(sorted(params_values), [2.0, 3.0, 4.0], rtol=0, atol=0)
    # Each point's report dict carries the pass verdict from the stub.
    for pt in bucket["points"]:
        assert pt["report"]["overall_verdict"] == "pass"


def test_rank_for_l8_prefers_layer6_then_layer3():
    """42.8a-fu1 (5): rank_for_l8 ranks by L6 tgb ascending when L6 is present, else L3 max_stab_eig."""
    # Helper: build a SweepPoint from a stub report.
    def _pt(param: float, *, verdict: str, l3_eig: float, l6_tgb: float | None):
        report = _make_stub_report(verdict=verdict, l3_eig=l3_eig, l6_tgb=l6_tgb)
        return brady2d_sweep.SweepPoint(
            param=param,
            params_dict={"sigma": param},
            report=report,
        )

    # Case A: both points have L6 → ranked by L6 tgb ascending.
    a = _pt(2.0, verdict="pass", l3_eig=-1e-3, l6_tgb=50.0)
    b = _pt(3.0, verdict="pass", l3_eig=-5e-4, l6_tgb=10.0)
    c = _pt(4.0, verdict="fail", l3_eig=+1.0, l6_tgb=2.0)  # failing — excluded
    ranked = brady2d_sweep.rank_for_l8([a, b, c], max_layer=6)
    assert [p.param for p in ranked] == [3.0, 2.0]  # tgb=10 first, then tgb=50
    assert all(p.verdict() == "pass" for p in ranked)

    # Case B: L6 missing on some passing points → fall back to L3 max_stab_eig ascending.
    d = _pt(5.0, verdict="pass", l3_eig=-1e-3, l6_tgb=None)
    e = _pt(6.0, verdict="pass", l3_eig=-5e-3, l6_tgb=None)
    ranked2 = brady2d_sweep.rank_for_l8([d, e], max_layer=3)
    assert [p.param for p in ranked2] == [6.0, 5.0]  # -5e-3 (more negative) first

    # Case C: no passing points → empty list.
    ranked3 = brady2d_sweep.rank_for_l8([c], max_layer=6)
    assert ranked3 == []

    # Case D: caps at TOP_K_FOR_L8.
    many = [
        _pt(float(i), verdict="pass", l3_eig=-float(i), l6_tgb=float(i))
        for i in range(1, 8)
    ]
    ranked4 = brady2d_sweep.rank_for_l8(many, max_layer=6)
    assert len(ranked4) == brady2d_sweep.TOP_K_FOR_L8


def test_rank_for_l8_shallow_max_layer_warns():
    """46.6a: rank_for_l8 emits UserWarning instead of silently degrading.

    When max_layer < 3 (or layer3 reports are absent at >=3), the previous
    behavior was to silently set the ranking key to a constant 0.0, producing
    arbitrary equal-rank ordering with no diagnostic. After 46.6a, this case
    must emit a UserWarning so it shows up in CI logs.
    """
    def _pt_no_layer3(param: float):
        from stencil_gen.brady2d_stability import StabilityReport

        report = StabilityReport.empty()
        report.layer1 = {"boundary_gv_err": 1e-3}
        report.overall_verdict = "pass"
        return brady2d_sweep.SweepPoint(
            param=param,
            params_dict={"sigma": param},
            report=report,
        )

    points = [_pt_no_layer3(1.0), _pt_no_layer3(2.0)]

    with pytest.warns(UserWarning, match="too shallow"):
        ranked = brady2d_sweep.rank_for_l8(points, max_layer=1)

    assert len(ranked) == 2
    assert {p.param for p in ranked} == {1.0, 2.0}


def test_report_to_dict_includes_layer_bl42():
    """45.6a.1.1: sweep copy of _report_to_dict must serialize layer_bl42."""
    from stencil_gen.brady2d_stability import StabilityReport

    r = StabilityReport(
        layer_bl42={"max_spectral_abscissa": 0.5},
        overall_verdict="pass",
    )
    out = brady2d_sweep._report_to_dict(r)
    assert "layer_bl42" in out
    assert out["layer_bl42"]["max_spectral_abscissa"] == 0.5


def test_report_to_dict_layer_bl42_filters_non_numeric():
    """45.6a.1.1: dict-valued sub-entries (spectral_abscissa_by_n) are dropped."""
    from stencil_gen.brady2d_stability import StabilityReport

    r = StabilityReport(
        layer_bl42={
            "spectral_abscissa_by_n": {21: 1e-12, 41: 2e-12},
            "max_spectral_abscissa": 2e-12,
            "purely_imaginary": True,
        },
        overall_verdict="pass",
    )
    out = brady2d_sweep._report_to_dict(r)
    assert "layer_bl42" in out
    assert "spectral_abscissa_by_n" not in out["layer_bl42"]
    assert out["layer_bl42"]["max_spectral_abscissa"] == pytest.approx(2e-12)
    assert out["layer_bl42"]["purely_imaginary"] == 1.0


def test_report_to_dict_omits_layer_bl42_when_none():
    """45.6a.1.1: key absent when layer_bl42 is None (no crash, no spurious entry)."""
    from stencil_gen.brady2d_stability import StabilityReport

    r = StabilityReport.empty()
    out = brady2d_sweep._report_to_dict(r)
    assert "layer_bl42" not in out
