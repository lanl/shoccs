"""Unit tests for :mod:`sweeps._bo_io` and :mod:`sweeps.bo`.

Covers:

- :class:`TestBOIO` — int-key restoration (plan 47.4c.1), file creation,
  filename-includes-seed, sorted iteration, and complex round-tripping
  through ``extras`` (plan 47.4d).
- :class:`TestBOCLI` — argparse surface (minimal accepting invocation,
  budget-mutex enforcement, cheap > HF rejection), dispatch through
  ``python -m sweeps bo --help``, and the ``--baseline staged`` stub
  acceptance (plan 47.4d).
- :class:`TestValidateWithCpp` — branch coverage for
  :func:`sweeps.bo._run_cpp_validation` (plan 47.5c): happy path, soft
  failure, all skip paths, the L8-raises capture path, the unconditional
  ``cpp_cutcell_violates_197_288`` recording for E4-classical, and the
  end-to-end "validate runs before persist" lifecycle.
- :class:`TestBaselineStaged` — failure-mode (47.5b.2) plus the remaining
  47.5c contract: ``--baseline`` omitted ⇒ no ``extras["baseline"]``;
  ``--baseline staged --persist`` ⇒ both winners on disk.

The :class:`TestBOCLI` tests stub :func:`run_mfbo` via ``monkeypatch`` so
no botorch / brady2d pipeline is entered — keeps the tests in the fast
suite.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from stencil_gen.bo import BOEval, BOResult, make_multi_fidelity_objective
from stencil_gen.brady2d_stability import L8_FINAL_LINF_TOL
from stencil_gen.optimizer import params_from_vector

from sweeps import bo as bo_cli
from sweeps._bo_io import (
    _INT_KEYED_TOP_LEVEL,
    iter_bo_runs,
    load_bo_run,
    save_bo_run,
)


def _make_bo_eval(*, fidelity: int = 1, value: float = 0.1) -> BOEval:
    return BOEval(
        x=np.array([-0.77, 0.16]),
        params={"alpha": [-0.77, 0.16]},
        fidelity=fidelity,
        value=value,
        wall_time=0.05,
        report={"failed_layer": None},
    )


def _make_bo_result(eval_history: tuple[BOEval, ...] = ()) -> BOResult:
    hf_history = tuple(e for e in eval_history if e.fidelity == 7)
    return BOResult(
        best_x=np.array([-0.77, 0.16]),
        best_params={"alpha": [-0.77, 0.16]},
        best_objective=0.42,
        best_report={"failed_layer": None},
        method="BoTorch-qMFKG",
        scheme="E4",
        kernel="classical",
        bounds=((-2.0, 2.0), (0.05, 2.0)),
        fidelity_levels=(1, 3, 7),
        hf_level=7,
        report_fields_by_layer={
            1: "layer1.boundary_gv_err",
            3: "layer3.max_stab_eig",
            7: "layer7.max_spectral_abscissa",
        },
        cost_model={1: 0.076, 3: 0.038, 7: 1.434},
        n_evals_per_fidelity={1: 9, 3: 3, 7: 2},
        wall_time_per_fidelity={1: 0.7, 3: 0.1, 7: 2.9},
        total_compute_time=3.7,
        eval_history=eval_history,
        hf_eval_history=hf_history,
        gp_hyperparameters={"lengthscale": [1.0, 1.0], "outputscale": 1.0},
        seed=1,
        converged=True,
        stop_reason="variance",
        extras={"n_sentinel_filtered": 0},
    )


class TestBOIO:
    """Plan 47.4c / 47.4c.1: per-run JSON persistence + int-key restoration."""

    def test_load_restores_int_keys(self, tmp_path):
        """Plan 47.4c.1: the four whitelisted dict[int,...] fields come back as int."""
        history = (_make_bo_eval(fidelity=1), _make_bo_eval(fidelity=7, value=0.42))
        result = _make_bo_result(history)
        path = save_bo_run(result, directory=tmp_path)
        loaded = load_bo_run(path)
        for name in _INT_KEYED_TOP_LEVEL:
            assert name in loaded, f"missing field {name} in loaded payload"
            for key in loaded[name].keys():
                assert isinstance(key, int), (
                    f"loaded {name} has non-int key {key!r} ({type(key).__name__})"
                )
        # Concrete spot-check against the source values.
        assert loaded["report_fields_by_layer"][7] == "layer7.max_spectral_abscissa"
        assert loaded["cost_model"][1] == pytest.approx(0.076)
        assert loaded["n_evals_per_fidelity"][3] == 3

    def test_roundtrip_preserves_eval_history(self, tmp_path):
        """Save + load: every BOEval field round-trips, int keys preserved."""
        history = (
            _make_bo_eval(fidelity=1, value=0.10),
            _make_bo_eval(fidelity=3, value=0.05),
            _make_bo_eval(fidelity=7, value=0.42),
        )
        result = _make_bo_result(history)
        path = save_bo_run(result, directory=tmp_path)
        loaded = load_bo_run(path)
        assert len(loaded["eval_history"]) == len(history)
        for src, dst in zip(history, loaded["eval_history"]):
            assert dst["fidelity"] == src.fidelity
            assert dst["value"] == pytest.approx(src.value)
            assert dst["wall_time"] == pytest.approx(src.wall_time)
            assert dst["x"] == pytest.approx(src.x.tolist())
            assert dst["params"] == src.params
            assert dst["report"] == src.report
        # Plan 47.4c.1 strengthening: int-keyed top-level fields equal the source dict.
        assert loaded["report_fields_by_layer"] == result.report_fields_by_layer
        assert loaded["cost_model"] == result.cost_model
        assert loaded["n_evals_per_fidelity"] == result.n_evals_per_fidelity
        assert loaded["wall_time_per_fidelity"] == result.wall_time_per_fidelity

    def test_make_objective_accepts_loaded_report_fields(self, tmp_path):
        """Plan 47.4c.1: loaded report_fields_by_layer flows into the factory.

        Without int-key restoration, ``make_multi_fidelity_objective`` raises
        ``TypeError`` at field-vs-layer validation (``int > str``).
        """
        result = _make_bo_result()
        path = save_bo_run(result, directory=tmp_path)
        loaded = load_bo_run(path)
        # Must not raise.
        objective = make_multi_fidelity_objective(
            loaded["scheme"],
            loaded["kernel"],
            loaded["report_fields_by_layer"],
        )
        assert callable(objective)

    def test_save_bo_run_creates_file(self, tmp_path):
        """``save_bo_run`` writes a JSON file at the documented filename."""
        result = _make_bo_result()
        path = save_bo_run(result, directory=tmp_path)
        assert path.exists()
        assert path.is_file()
        # Filename schema: {scheme}_{kernel}_{mangled_HF_field}_{seed}.json.
        # HF field is layer7.max_spectral_abscissa → layer7_max_spectral_abscissa.
        assert path.name == "E4_classical_layer7_max_spectral_abscissa_1.json"
        assert path.parent == tmp_path
        # Payload is valid JSON with the documented top-level keys.
        payload = json.loads(path.read_text())
        for required in ("best_x", "best_objective", "scheme", "kernel",
                         "fidelity_levels", "eval_history"):
            assert required in payload, f"missing top-level key {required!r}"

    def test_serializer_handles_complex(self, tmp_path):
        """``_BOEncoder`` serialises ``complex`` values as ``[real, imag]``.

        Any layer that surfaces a :class:`KreissResult` (``layer2`` if it
        lands in fidelity_layers — see plan 46.2b) carries a ``witness_s``
        complex through ``extras``.  Without the encoder branch
        ``json.dump`` would raise ``TypeError``.
        """
        result = _make_bo_result()
        # Inject a complex in extras and rebuild a frozen dataclass with it.
        # ``BOResult`` is frozen, so we must build a fresh instance.
        from dataclasses import replace

        result_with_complex = replace(
            result,
            extras={"witness_s": 1.5 + 2.5j, "n_sentinel_filtered": 0},
        )
        path = save_bo_run(result_with_complex, directory=tmp_path)
        payload = json.loads(path.read_text())
        # Complex should round-trip as a 2-element list of floats.
        assert payload["extras"]["witness_s"] == [1.5, 2.5]

    def test_filename_includes_seed(self, tmp_path):
        """Same (scheme, kernel, objective) but distinct seeds → distinct files."""
        from dataclasses import replace

        base = _make_bo_result()
        r1 = replace(base, seed=1)
        r2 = replace(base, seed=2)
        p1 = save_bo_run(r1, directory=tmp_path)
        p2 = save_bo_run(r2, directory=tmp_path)
        assert p1 != p2
        assert p1.name.endswith("_1.json")
        assert p2.name.endswith("_2.json")
        # Both files exist and are independent.
        assert p1.exists() and p2.exists()
        assert json.loads(p1.read_text())["seed"] == 1
        assert json.loads(p2.read_text())["seed"] == 2

    def test_iter_bo_runs_sorted(self, tmp_path):
        """``iter_bo_runs`` yields ``*.json`` paths in sorted order."""
        from dataclasses import replace

        base = _make_bo_result()
        # Save in deliberately non-sorted order to exercise the sort.
        save_bo_run(replace(base, seed=3), directory=tmp_path)
        save_bo_run(replace(base, seed=1), directory=tmp_path)
        save_bo_run(replace(base, seed=2), directory=tmp_path)
        # Drop a non-JSON file to confirm it is ignored.
        (tmp_path / "ignore.txt").write_text("not json\n")

        seen = list(iter_bo_runs(tmp_path))
        assert all(p.suffix == ".json" for p in seen)
        assert seen == sorted(seen)
        assert [p.name for p in seen] == [
            "E4_classical_layer7_max_spectral_abscissa_1.json",
            "E4_classical_layer7_max_spectral_abscissa_2.json",
            "E4_classical_layer7_max_spectral_abscissa_3.json",
        ]


# ---------------------------------------------------------------------------
# TestBOCLI — sweeps.bo argparse + dispatch (plan 47.4d)
# ---------------------------------------------------------------------------


class TestBOCLI:
    """Argparse surface and dispatch wiring for ``sweeps bo``."""

    def test_argparse_minimal_invocation(self, monkeypatch, capsys):
        """Minimal accepting invocation routes (parsed args) into run_mfbo."""
        calls: list[dict] = []

        def _fake_run_mfbo(**kwargs):
            calls.append(kwargs)
            # Return a stub BOResult shaped to match the (cheap=1, HF=3) call.
            stub = _make_bo_result()
            from dataclasses import replace
            return replace(
                stub,
                fidelity_levels=(1, 3),
                hf_level=3,
                report_fields_by_layer={
                    1: "layer1.boundary_gv_err",
                    3: "layer3.max_stab_eig",
                },
                cost_model={1: 0.076, 3: 0.038},
                n_evals_per_fidelity={1: 5, 3: 5},
                wall_time_per_fidelity={1: 0.4, 3: 0.2},
                bounds=((0.5, 20.0),),
                kernel="tension",
            )

        monkeypatch.setattr(bo_cli, "run_mfbo", _fake_run_mfbo)

        rc = bo_cli.main(
            [
                "--scheme", "E4",
                "--kernel", "tension",
                "--objective", "layer3.max_stab_eig",
                "--cheap-fidelities", "1",
                "--bounds", "0.5", "20",
                "--budget-evals", "10",
                "--seed", "1",
            ]
        )
        assert rc == 0
        assert len(calls) == 1
        call = calls[0]
        assert call["scheme"] == "E4"
        assert call["kernel"] == "tension"
        assert call["seed"] == 1
        assert call["budget_evals"] == 10
        assert call["budget_seconds"] is None
        # report_fields_by_layer was assembled from --objective + --cheap-fidelities.
        assert call["report_fields_by_layer"] == {
            1: "layer1.boundary_gv_err",
            3: "layer3.max_stab_eig",
        }
        # 1D bounds for the tension kernel.
        assert list(call["bounds"]) == [(0.5, 20.0)]
        out = capsys.readouterr().out
        assert "BoTorch-qMFKG" in out
        assert "layer3.max_stab_eig" in out

    def test_argparse_rejects_no_budget(self):
        """Neither --budget-evals nor --budget-seconds → SystemExit (mutex required)."""
        with pytest.raises(SystemExit) as exc_info:
            bo_cli.main(
                [
                    "--scheme", "E4",
                    "--kernel", "tension",
                    "--objective", "layer3.max_stab_eig",
                    "--cheap-fidelities", "1",
                    "--bounds", "0.5", "20",
                ]
            )
        assert exc_info.value.code != 0

    def test_argparse_rejects_both_budgets(self):
        """Both --budget-evals and --budget-seconds → SystemExit (mutex)."""
        with pytest.raises(SystemExit) as exc_info:
            bo_cli.main(
                [
                    "--scheme", "E4",
                    "--kernel", "tension",
                    "--objective", "layer3.max_stab_eig",
                    "--cheap-fidelities", "1",
                    "--bounds", "0.5", "20",
                    "--budget-evals", "10",
                    "--budget-seconds", "5.0",
                ]
            )
        assert exc_info.value.code != 0

    def test_argparse_rejects_bad_field_layer(self, monkeypatch):
        """A cheap fidelity ≥ HF layer → parser.error (cheap > HF)."""
        # If the parser passes the bad spec through, run_mfbo would be
        # called.  Sentinel here detects that case and surfaces it.
        def _unexpected(**kwargs):
            raise AssertionError(
                "run_mfbo should not be invoked when cheap-fidelities >= HF"
            )

        monkeypatch.setattr(bo_cli, "run_mfbo", _unexpected)

        with pytest.raises(SystemExit) as exc_info:
            bo_cli.main(
                [
                    "--scheme", "E4",
                    "--kernel", "classical",
                    "--objective", "layer7.max_spectral_abscissa",
                    "--cheap-fidelities", "8",  # > HF inferred from layer7.*
                    "--budget-evals", "10",
                ]
            )
        assert exc_info.value.code != 0

    def test_dispatch_via_main(self):
        """``python -m sweeps bo --help`` exits 0 and lists the BO flags."""
        stencil_gen_dir = Path(__file__).resolve().parent.parent
        env = os.environ.copy()
        env.setdefault("SYMPY_CACHE_SIZE", "50000")
        proc = subprocess.run(
            [sys.executable, "-m", "sweeps", "bo", "--help"],
            cwd=str(stencil_gen_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, (
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
        for needle in (
            "--objective",
            "--cheap-fidelities",
            "--budget-evals",
            "--budget-seconds",
            "--baseline",
            "--validate-with-cpp",
            "--persist",
        ):
            assert needle in proc.stdout, (
                f"missing {needle!r} in `python -m sweeps bo --help` output"
            )

    def test_baseline_staged_invokes_run_staged_optimize(self, monkeypatch, capsys):
        """Plan 47.5b: ``--baseline staged`` actually runs ``run_staged_optimize``.

        Both ``run_mfbo`` and ``run_staged_optimize`` are monkeypatched so no
        cascade is invoked.  The test pins (a) ``run_staged_optimize`` is called,
        (b) the same seed flows into both methods (fairness), (c) the
        ``best_objective`` HF objective field flows through verbatim, and (d) the
        baseline record lands under ``result.extras["baseline"]`` and in the
        side-by-side print.

        Per the plan-46 lesson cited in 47.5a, ``--baseline staged`` must NOT
        emit the legacy "deferred" message; that path is removed in 47.5b.
        """
        from dataclasses import replace

        bo_calls: list[dict] = []
        staged_calls: list[dict] = []

        def _fake_run_mfbo(**kwargs):
            bo_calls.append(kwargs)
            stub = _make_bo_result()
            return replace(
                stub,
                fidelity_levels=(1, 3),
                hf_level=3,
                report_fields_by_layer={
                    1: "layer1.boundary_gv_err",
                    3: "layer3.max_stab_eig",
                },
                cost_model={1: 0.076, 3: 0.038},
                n_evals_per_fidelity={1: 5, 3: 5},
                wall_time_per_fidelity={1: 0.4, 3: 0.2},
                bounds=((0.5, 20.0),),
                kernel="tension",
                best_x=np.array([10.0]),
                best_params={"sigma": 10.0},
                best_objective=-1.0e-3,
                extras={"n_sentinel_filtered": 0},
            )

        def _fake_run_staged_optimize(**kwargs):
            staged_calls.append(kwargs)
            from stencil_gen.optimizer import OptimizeResult as _OR

            return _OR(
                best_params={"sigma": 11.0},
                best_x=np.array([11.0]),
                best_objective=-2.0e-3,
                best_report={"failed_layer": None},
                method="staged",
                converged=True,
                n_evals=42,
                compute_time=1.5,
                history=[],
                extras={
                    "stage": "validated",
                    "validator_ranking": [(np.array([11.0]), -2.0e-3), (np.array([12.0]), -1.5e-3)],
                },
            )

        monkeypatch.setattr(bo_cli, "run_mfbo", _fake_run_mfbo)
        monkeypatch.setattr(bo_cli, "run_staged_optimize", _fake_run_staged_optimize)

        rc = bo_cli.main(
            [
                "--scheme", "E4",
                "--kernel", "tension",
                "--objective", "layer3.max_stab_eig",
                "--cheap-fidelities", "1",
                "--bounds", "0.5", "20",
                "--budget-evals", "10",
                "--seed", "7",
                "--baseline", "staged",
            ]
        )
        assert rc == 0
        assert len(bo_calls) == 1
        assert len(staged_calls) == 1
        # Both runs share the seed.
        assert bo_calls[0]["seed"] == 7
        assert staged_calls[0]["seed"] == 7
        # Staged reuses the HF objective field that MF-BO targeted.
        assert staged_calls[0]["report_field"] == "layer3.max_stab_eig"
        # Staged reuses the same bounds.
        assert staged_calls[0]["bounds"] == [(0.5, 20.0)]
        # Plan-body literal: n_restarts=10 for the baseline.
        assert staged_calls[0]["n_restarts"] == 10
        # Plan 47.5b.1.1 (fix branch 1): the BO baseline must match
        # ``python -m sweeps optimize --method staged ...``'s CLI-resolved
        # defaults (the user-facing reference) rather than
        # ``run_staged_optimize``'s function-level defaults.  CLI form is
        # ``inner_max_layer=3``, ``inner_gate=max(3-1,0)=2``.  HF=L3 here ⇒
        # ``validator_max_layer=3`` (``max(hf_level, 3)`` fairness fix).
        assert staged_calls[0]["inner_gate"] == 2
        assert staged_calls[0]["inner_max_layer"] == 3
        assert staged_calls[0]["validator_max_layer"] == 3
        out = capsys.readouterr().out
        # New behaviour: side-by-side comparison appears.
        assert "comparison (side-by-side)" in out
        assert "staged" in out
        # Old stub message must NOT appear post-47.5b.
        assert "deferred" not in out

    def test_baseline_staged_uses_canonical_inner_gate_at_hf_l7(
        self, monkeypatch, capsys
    ):
        """Plan 47.5b.1.1: with HF=L7, the BO baseline matches the
        ``sweeps optimize --method staged`` CLI defaults (``inner_gate=2``,
        ``inner_max_layer=3``) and lifts ``validator_max_layer`` to 7
        (fairness fix tracking MF-BO's HF target).

        This is the second half of the 47.5b.1.1 contract: fairness-fix
        raises validator depth; the CLI-default inner-stage gate is
        passed explicitly so the baseline is what users actually invoke.
        """
        from dataclasses import replace

        bo_calls: list[dict] = []
        staged_calls: list[dict] = []

        def _fake_run_mfbo(**kwargs):
            bo_calls.append(kwargs)
            stub = _make_bo_result()
            return replace(
                stub,
                fidelity_levels=(1, 3, 5, 6, 7),
                hf_level=7,
                report_fields_by_layer={
                    1: "layer1.boundary_gv_err",
                    3: "layer3.max_stab_eig",
                    5: "layer_bl42.max_spectral_abscissa",
                    6: "layer6.transient_growth_bound",
                    7: "layer7.max_spectral_abscissa",
                },
                cost_model={1: 0.076, 3: 0.038, 5: 0.486, 6: 0.846, 7: 1.434},
                n_evals_per_fidelity={1: 5, 3: 3, 5: 1, 6: 1, 7: 2},
                wall_time_per_fidelity={1: 0.4, 3: 0.1, 5: 0.5, 6: 0.8, 7: 2.9},
                bounds=((-2.0, 2.0), (0.05, 2.0)),
                kernel="classical",
                best_x=np.array([-0.7733, 0.1624]),
                best_params={"alpha": [-0.7733, 0.1624]},
                best_objective=-1.0e-3,
                extras={"n_sentinel_filtered": 0},
            )

        def _fake_run_staged_optimize(**kwargs):
            staged_calls.append(kwargs)
            from stencil_gen.optimizer import OptimizeResult as _OR

            return _OR(
                best_params={"alpha": [-0.7733, 0.1624]},
                best_x=np.array([-0.7733, 0.1624]),
                best_objective=-2.0e-3,
                best_report={"failed_layer": None},
                method="staged",
                converged=True,
                n_evals=42,
                compute_time=1.5,
                history=[],
                extras={
                    "stage": "validated",
                    "validator_ranking": [
                        (np.array([-0.7733, 0.1624]), -2.0e-3),
                    ],
                },
            )

        monkeypatch.setattr(bo_cli, "run_mfbo", _fake_run_mfbo)
        monkeypatch.setattr(bo_cli, "run_staged_optimize", _fake_run_staged_optimize)

        rc = bo_cli.main(
            [
                "--scheme", "E4",
                "--kernel", "classical",
                "--objective", "layer7.max_spectral_abscissa",
                "--cheap-fidelities", "1", "3", "5", "6",
                "--bounds", "-2", "2", "0.05", "2",
                "--budget-evals", "12",
                "--seed", "1",
                "--baseline", "staged",
            ]
        )
        assert rc == 0
        assert len(staged_calls) == 1
        # Fairness fix: validator depth tracks MF-BO's HF target.
        assert staged_calls[0]["validator_max_layer"] == 7
        # CLI-default inner-stage gate (plan 47.5b.1.1, fix branch 1):
        # ``inner_max_layer = min(3, validator_max_layer) = 3``,
        # ``inner_gate = max(inner_max_layer - 1, 0) = 2``.  Same values
        # for HF=L3 and HF=L7 because ``min(3, ...)`` saturates at 3.
        assert staged_calls[0]["inner_gate"] == 2
        assert staged_calls[0]["inner_max_layer"] == 3
        capsys.readouterr()


# ---------------------------------------------------------------------------
# TestBaselineStaged — failure-mode coverage for ``--baseline staged`` (47.5b.2)
# ---------------------------------------------------------------------------


class TestBaselineStaged:
    """Plan 47.5b.2: a baseline failure must not lose the MF-BO investment.

    Pre-fix, an exception from ``run_staged_optimize`` propagated past the
    summary-print and ``--persist`` blocks, discarding the ~5 minutes of
    BO wall-time the user just invested.  These tests pin the post-fix
    contract: the failure is recorded under
    ``result.extras["baseline"]["error"]``, the run continues to print and
    persist, and the summary table renders cleanly with ``None``-valued
    numeric fields.
    """

    def test_baseline_failure_does_not_lose_persistence(
        self, monkeypatch, capsys, tmp_path
    ):
        """Plan 47.5b.2: ``run_staged_optimize`` raising still leaves a
        persisted MF-BO JSON on disk with the baseline failure recorded."""
        from dataclasses import replace

        history = (
            _make_bo_eval(fidelity=1, value=0.10),
            _make_bo_eval(fidelity=3, value=-1.0e-3),
        )

        def _fake_run_mfbo(**kwargs):
            stub = _make_bo_result(history)
            return replace(
                stub,
                fidelity_levels=(1, 3),
                hf_level=3,
                report_fields_by_layer={
                    1: "layer1.boundary_gv_err",
                    3: "layer3.max_stab_eig",
                },
                cost_model={1: 0.076, 3: 0.038},
                n_evals_per_fidelity={1: 5, 3: 5},
                wall_time_per_fidelity={1: 0.4, 3: 0.2},
                bounds=((0.5, 20.0),),
                kernel="tension",
                best_x=np.array([10.0]),
                best_params={"sigma": 10.0},
                best_objective=-1.0e-3,
                extras={"n_sentinel_filtered": 0},
            )

        def _boom(**kwargs):
            raise RuntimeError("synthetic baseline boom")

        # Redirect the persist target so the test does not write into the
        # repo's ``sweeps/bo_runs/`` directory.  Patching ``BO_RUNS_DIR`` is
        # not sufficient because ``save_bo_run`` binds its ``directory``
        # default at function-definition time; wrap the call instead.
        import sweeps._bo_io as bo_io_mod

        original_save = bo_io_mod.save_bo_run

        def _save_to_tmp(result, directory=tmp_path):
            return original_save(result, directory=directory)

        monkeypatch.setattr(bo_cli, "run_mfbo", _fake_run_mfbo)
        monkeypatch.setattr(bo_cli, "run_staged_optimize", _boom)
        monkeypatch.setattr(bo_io_mod, "save_bo_run", _save_to_tmp)

        rc = bo_cli.main(
            [
                "--scheme", "E4",
                "--kernel", "tension",
                "--objective", "layer3.max_stab_eig",
                "--cheap-fidelities", "1",
                "--bounds", "0.5", "20",
                "--budget-evals", "10",
                "--seed", "1",
                "--baseline", "staged",
                "--persist",
            ]
        )
        # (a) run completes despite the baseline failure.
        assert rc == 0

        # (b) persisted JSON exists.
        persisted = tmp_path / "E4_tension_layer3_max_stab_eig_1.json"
        assert persisted.exists(), (
            f"persisted file missing; tmp_path contents: {list(tmp_path.iterdir())}"
        )

        # (c) extras.baseline.error captures the exception message.
        payload = json.loads(persisted.read_text())
        baseline = payload["extras"]["baseline"]
        assert "RuntimeError" in baseline["error"]
        assert "synthetic baseline boom" in baseline["error"]
        assert baseline["method"] == "staged"

        # (d) MF-BO fields are present and finite (the BO investment is preserved).
        assert "best_x" in payload
        assert payload["best_x"] == [10.0]
        assert payload["best_objective"] == pytest.approx(-1.0e-3)
        assert len(payload["eval_history"]) > 0

        # (e) stdout warns about the failure but does not raise.
        out = capsys.readouterr().out
        assert "--baseline staged: run_staged_optimize raised" in out
        assert "RuntimeError" in out

    def test_baseline_failure_print_summary_renders(self, monkeypatch, capsys):
        """Plan 47.5b.2: ``_print_summary`` survives a None-valued baseline_record.

        The failure-sentinel dict has ``best_objective=None`` and
        ``compute_time=None``; the ``isinstance(sg_obj, (int, float))``
        branch on ``bo.py:271`` is the only thing keeping the summary
        formatting from raising ``TypeError`` on the missing numerics.
        """
        from dataclasses import replace

        def _fake_run_mfbo(**kwargs):
            stub = _make_bo_result()
            return replace(
                stub,
                fidelity_levels=(1, 3),
                hf_level=3,
                report_fields_by_layer={
                    1: "layer1.boundary_gv_err",
                    3: "layer3.max_stab_eig",
                },
                cost_model={1: 0.076, 3: 0.038},
                n_evals_per_fidelity={1: 5, 3: 5},
                wall_time_per_fidelity={1: 0.4, 3: 0.2},
                bounds=((0.5, 20.0),),
                kernel="tension",
                best_x=np.array([10.0]),
                best_params={"sigma": 10.0},
                best_objective=-1.0e-3,
                extras={"n_sentinel_filtered": 0},
            )

        def _boom(**kwargs):
            raise ValueError("synthetic boom for summary")

        monkeypatch.setattr(bo_cli, "run_mfbo", _fake_run_mfbo)
        monkeypatch.setattr(bo_cli, "run_staged_optimize", _boom)

        rc = bo_cli.main(
            [
                "--scheme", "E4",
                "--kernel", "tension",
                "--objective", "layer3.max_stab_eig",
                "--cheap-fidelities", "1",
                "--bounds", "0.5", "20",
                "--budget-evals", "10",
                "--seed", "1",
                "--baseline", "staged",
            ]
        )
        assert rc == 0

        out = capsys.readouterr().out
        # Side-by-side comparison still rendered (None coerces to "None"
        # via the ``str(sg_obj)`` else-branch).
        assert "comparison (side-by-side)" in out
        assert "staged" in out
        # The MF-BO row is still well-formed in scientific notation.
        assert "BoTorch-qMFKG" in out
        # Failure breadcrumb visible.
        assert "synthetic boom for summary" in out


# ---------------------------------------------------------------------------
# Helpers shared by TestValidateWithCpp and TestBaselineStaged (plan 47.5c)
# ---------------------------------------------------------------------------


def _fake_layer8_report(*, stable: bool, final_linf: float, wall_time_s: float):
    """Stub return value for a monkeypatched ``brady2d_stability_score``.

    ``_run_cpp_validation`` only reads ``report.layer8`` (a dict) — return
    a ``SimpleNamespace`` with that one field so the tests do not need to
    construct a full :class:`StabilityReport`.
    """
    return types.SimpleNamespace(
        layer8={
            "stable": bool(stable),
            "final_linf": float(final_linf),
            "wall_time_s": float(wall_time_s),
        }
    )


def _make_classical_result(
    *,
    scheme: str = "E4",
    kernel: str = "classical",
    best_x: np.ndarray | None = None,
    best_objective: float = -1.0e-3,
) -> BOResult:
    """Build a synthetic ``BOResult`` with kernel-correct ``best_params``.

    Uses :func:`params_from_vector` so the ``best_params`` shape matches
    what :func:`run_mfbo` emits in production — the 47.5a Done note flagged
    a smoke-test bug from passing the wrong shape.  Shape:

    - ``classical``  : ``best_x = [α₀, α₁]``, ``best_params = {"alpha": [α₀, α₁]}``
    - ``tension``    : ``best_x = [σ]``,      ``best_params = {"sigma": σ}``
    """
    from dataclasses import replace

    if best_x is None:
        best_x = (
            np.array([-0.7733, 0.1624])
            if kernel == "classical"
            else np.array([10.0])
        )
    bounds = (
        ((-2.0, 2.0), (0.05, 2.0))
        if kernel == "classical"
        else ((0.5, 20.0),)
    )
    base = _make_bo_result()
    return replace(
        base,
        scheme=scheme,
        kernel=kernel,
        best_x=np.asarray(best_x, dtype=float),
        best_params=params_from_vector(kernel, np.asarray(best_x, dtype=float)),
        best_objective=best_objective,
        bounds=bounds,
    )


# ---------------------------------------------------------------------------
# TestValidateWithCpp — branch coverage for _run_cpp_validation (plan 47.5c)
# ---------------------------------------------------------------------------


class TestValidateWithCpp:
    """Plan 47.5c: pin the branch surface of :func:`_run_cpp_validation`.

    The 47.5a smoke tests covered only the skip paths and the L8-raises
    path; three real branches were uncovered: the L8-PASS print, the
    soft-failure (stable=True, final_linf > tol) print, and the
    unconditional ``cpp_cutcell_violates_197_288`` record for
    E4-classical.  All tests here patch ``sweeps.bo.brady2d_stability_score``
    rather than the upstream symbol so the patch is module-local, and they
    construct ``best_params`` via :func:`params_from_vector` so the
    codified path matches what :func:`run_mfbo` actually produces.
    """

    def test_happy_path_records_full_schema(self, monkeypatch, capsys):
        """L8 PASS branch: stable=True, final_linf below tolerance.

        Schema for a non-classical kernel: exactly the three keys
        ``{l8_stable, l8_final_linf, wall_time_s}``.  No ``l8_error`` (no
        exception) and no ``cpp_cutcell_violates_197_288`` (kernel != classical
        keeps :func:`_record_cpp_cutcell_diagnostic` from writing the flag).
        """
        monkeypatch.setattr(
            bo_cli,
            "brady2d_stability_score",
            lambda *a, **kw: _fake_layer8_report(
                stable=True, final_linf=0.001, wall_time_s=1.5
            ),
        )
        result = _make_classical_result(
            kernel="tension", best_x=np.array([10.0])
        )
        entry = bo_cli._run_cpp_validation(result)
        assert entry is not None
        assert set(entry.keys()) == {"l8_stable", "l8_final_linf", "wall_time_s"}
        assert entry["l8_stable"] is True
        assert entry["l8_final_linf"] == pytest.approx(0.001)
        assert entry["wall_time_s"] == pytest.approx(1.5)
        out = capsys.readouterr().out
        assert "L8 PASS" in out

    def test_soft_failure_branch_records_stable_true_but_warns(
        self, monkeypatch, capsys
    ):
        """L8 soft-failure: ``stable=True`` but ``final_linf > L8_FINAL_LINF_TOL``.

        The verdict ``stable=True`` is preserved verbatim (the C++ stack
        signalled stable); the warning ``WARNING: L8 soft-failure`` fires
        on a separate code path from the L8-PASS and L8-FAIL prints.
        """
        soft_linf = L8_FINAL_LINF_TOL + 0.5
        monkeypatch.setattr(
            bo_cli,
            "brady2d_stability_score",
            lambda *a, **kw: _fake_layer8_report(
                stable=True, final_linf=soft_linf, wall_time_s=2.0
            ),
        )
        result = _make_classical_result(
            kernel="tension", best_x=np.array([10.0])
        )
        entry = bo_cli._run_cpp_validation(result)
        assert entry is not None
        assert entry["l8_stable"] is True
        assert entry["l8_final_linf"] > L8_FINAL_LINF_TOL
        out = capsys.readouterr().out
        assert "WARNING: L8 soft-failure" in out

    def test_validate_runs_before_persist(self, monkeypatch, capsys, tmp_path):
        """End-to-end via ``main(...)``: ``--validate-with-cpp`` mutates
        ``result.extras`` BEFORE ``--persist`` writes the JSON.

        Pre-fix to plan 45.5a.1's lesson, swapping the order would persist
        a stale snapshot that omits the L8 verdict.  This test pins the
        order by reading the on-disk JSON: the ``cpp_validation`` payload
        must land in the file, not just on the in-memory result.
        """
        from dataclasses import replace

        def _fake_run_mfbo(**kwargs):
            stub = _make_classical_result(
                kernel="tension", best_x=np.array([10.0])
            )
            return replace(
                stub,
                fidelity_levels=(1, 3),
                hf_level=3,
                report_fields_by_layer={
                    1: "layer1.boundary_gv_err",
                    3: "layer3.max_stab_eig",
                },
                cost_model={1: 0.076, 3: 0.038},
                n_evals_per_fidelity={1: 5, 3: 5},
                wall_time_per_fidelity={1: 0.4, 3: 0.2},
                extras={"n_sentinel_filtered": 0},
            )

        # Redirect persistence to ``tmp_path`` (default-binding caveat from
        # 47.5b.2: wrap save_bo_run instead of patching BO_RUNS_DIR).
        import sweeps._bo_io as bo_io_mod

        original_save = bo_io_mod.save_bo_run

        def _save_to_tmp(result, directory=tmp_path):
            return original_save(result, directory=directory)

        monkeypatch.setattr(bo_cli, "run_mfbo", _fake_run_mfbo)
        monkeypatch.setattr(
            bo_cli,
            "brady2d_stability_score",
            lambda *a, **kw: _fake_layer8_report(
                stable=True, final_linf=0.002, wall_time_s=1.7
            ),
        )
        # Default ``SHOCCS_BINARY`` may not exist in the test sandbox —
        # patch ``.exists`` to True so the validate path runs.
        monkeypatch.setattr(
            bo_cli,
            "SHOCCS_BINARY",
            types.SimpleNamespace(exists=lambda: True),
        )
        monkeypatch.setattr(bo_io_mod, "save_bo_run", _save_to_tmp)

        rc = bo_cli.main(
            [
                "--scheme", "E4",
                "--kernel", "tension",
                "--objective", "layer3.max_stab_eig",
                "--cheap-fidelities", "1",
                "--bounds", "0.5", "20",
                "--budget-evals", "10",
                "--seed", "1",
                "--validate-with-cpp",
                "--persist",
            ]
        )
        assert rc == 0

        persisted = tmp_path / "E4_tension_layer3_max_stab_eig_1.json"
        assert persisted.exists(), (
            f"expected persisted JSON at {persisted}; "
            f"tmp_path contents: {list(tmp_path.iterdir())}"
        )
        payload = json.loads(persisted.read_text())
        validation = payload["extras"]["cpp_validation"]
        assert validation["l8_stable"] is True
        assert validation["l8_final_linf"] == pytest.approx(0.002)
        assert validation["wall_time_s"] == pytest.approx(1.7)
        out = capsys.readouterr().out
        assert "L8 PASS" in out

    def test_skips_on_unsupported_kernel(self, capsys):
        """``kernel=tension-penalty`` (not in ``_CPP_SUPPORTED_KERNELS``) ⇒
        validation returns ``None`` and ``extras`` is left untouched."""
        from dataclasses import replace

        result = _make_classical_result(kernel="classical")
        # Force an out-of-table kernel via dataclass-replace (BOResult does
        # not validate the kernel string at construction time).
        result = replace(result, kernel="tension-penalty")
        original_extras = dict(result.extras)
        entry = bo_cli._run_cpp_validation(result)
        assert entry is None
        assert dict(result.extras) == original_extras
        out = capsys.readouterr().out
        assert "is not C++-supported" in out

    def test_skips_on_unsupported_scheme(self, capsys):
        """``scheme=E2`` (no L8 bridge wired) ⇒ validation returns
        ``None`` and prints the documented skip message."""
        from dataclasses import replace

        result = _make_classical_result(kernel="classical")
        result = replace(result, scheme="E2")
        entry = bo_cli._run_cpp_validation(result)
        assert entry is None
        out = capsys.readouterr().out
        assert "no L8 bridge wired" in out

    def test_skips_on_empty_best_params(self, capsys):
        """``best_params = {}`` ⇒ skip; the kernel skip does not fire first
        (the empty-params guard runs before the kernel check)."""
        from dataclasses import replace

        result = _make_classical_result(kernel="classical")
        result = replace(result, best_params={})
        entry = bo_cli._run_cpp_validation(result)
        assert entry is None
        out = capsys.readouterr().out
        assert "best_params is empty" in out

    def test_skips_on_nonfinite_best_objective(self, capsys):
        """``best_objective = inf`` ⇒ skip with the "non-finite" message."""
        from dataclasses import replace

        result = _make_classical_result(kernel="classical")
        result = replace(result, best_objective=float("inf"))
        entry = bo_cli._run_cpp_validation(result)
        assert entry is None
        out = capsys.readouterr().out
        assert "non-finite" in out

    def test_skips_on_missing_shoccs_binary(self, monkeypatch, capsys):
        """``SHOCCS_BINARY.exists() is False`` ⇒ skip with the build-hint
        message.  Patches the binding rather than the underlying ``Path``
        method so the patch reverts cleanly under ``monkeypatch``."""
        monkeypatch.setattr(
            bo_cli,
            "SHOCCS_BINARY",
            types.SimpleNamespace(exists=lambda: False),
        )
        result = _make_classical_result(kernel="classical")
        entry = bo_cli._run_cpp_validation(result)
        assert entry is None
        out = capsys.readouterr().out
        assert "shoccs binary not found" in out

    def test_records_l8_failure_not_raises(self, monkeypatch, capsys):
        """``brady2d_stability_score`` raising ⇒ ``_run_cpp_validation``
        returns a sentinel dict, does NOT propagate the exception.

        The 47.5a Done note pinned the schema: ``l8_stable=False``,
        ``l8_final_linf`` is ``NaN``, ``wall_time_s=0.0``, plus
        ``l8_error`` (NOT bare ``error``) carrying the typed message.
        """
        def _boom(*args, **kwargs):
            raise RuntimeError("synthetic L8 boom")

        monkeypatch.setattr(bo_cli, "brady2d_stability_score", _boom)
        # Bypass the binary-not-found skip so the L8 call actually fires.
        monkeypatch.setattr(
            bo_cli,
            "SHOCCS_BINARY",
            types.SimpleNamespace(exists=lambda: True),
        )
        result = _make_classical_result(
            kernel="tension", best_x=np.array([10.0])
        )
        entry = bo_cli._run_cpp_validation(result)
        assert entry is not None
        assert entry["l8_stable"] is False
        assert math.isnan(entry["l8_final_linf"])
        assert entry["wall_time_s"] == 0.0
        assert entry["l8_error"] == "RuntimeError: synthetic L8 boom"
        out = capsys.readouterr().out
        assert "L8 raised" in out

    @pytest.mark.parametrize(
        "scheme,kernel,best_x,expected_flag",
        [
            # Below the cut-cell floor (197/288 ≈ 0.6840) ⇒ flag True.
            ("E4", "classical", np.array([-0.7733, 0.1624]), True),
            # Above the cut-cell floor ⇒ flag False (still recorded).
            ("E4", "classical", np.array([-0.7733, 0.7]), False),
            # Non-classical kernel ⇒ flag absent entirely (per 47.5a Done note).
            ("E4", "tension", np.array([10.0]), None),
        ],
    )
    def test_cpp_cutcell_flag_recorded_for_e4_classical(
        self,
        monkeypatch,
        scheme,
        kernel,
        best_x,
        expected_flag,
    ):
        """Plan 47.5a Done note: ``cpp_cutcell_violates_197_288`` is
        recorded unconditionally for E4-classical (regardless of whether
        L8 actually ran), and is absent for non-classical kernels."""
        # Patch the L8 call to a benign return so the test does not actually
        # invoke C++ — the cut-cell flag is set BEFORE the brady2d call so
        # both paths exercise the unconditional record.
        monkeypatch.setattr(
            bo_cli,
            "brady2d_stability_score",
            lambda *a, **kw: _fake_layer8_report(
                stable=True, final_linf=0.001, wall_time_s=1.5
            ),
        )
        monkeypatch.setattr(
            bo_cli,
            "SHOCCS_BINARY",
            types.SimpleNamespace(exists=lambda: True),
        )
        result = _make_classical_result(
            scheme=scheme, kernel=kernel, best_x=best_x
        )
        entry = bo_cli._run_cpp_validation(result)
        assert entry is not None
        if expected_flag is None:
            assert "cpp_cutcell_violates_197_288" not in entry
        else:
            assert entry["cpp_cutcell_violates_197_288"] is expected_flag

    def test_cpp_cutcell_flag_present_when_l8_raises(
        self, monkeypatch, capsys
    ):
        """Combine the L8-raises path with E4-classical at α₁ < 197/288:
        the cut-cell flag must appear in the returned dict alongside the
        ``l8_error`` breadcrumb (the flag is recorded BEFORE the L8 call,
        so a raising L8 cannot suppress it)."""
        def _boom(*args, **kwargs):
            raise RuntimeError("synthetic L8 boom")

        monkeypatch.setattr(bo_cli, "brady2d_stability_score", _boom)
        monkeypatch.setattr(
            bo_cli,
            "SHOCCS_BINARY",
            types.SimpleNamespace(exists=lambda: True),
        )
        result = _make_classical_result(
            scheme="E4",
            kernel="classical",
            best_x=np.array([-0.7733, 0.1624]),
        )
        entry = bo_cli._run_cpp_validation(result)
        assert entry is not None
        assert entry["cpp_cutcell_violates_197_288"] is True
        assert entry["l8_error"] == "RuntimeError: synthetic L8 boom"
        assert entry["l8_stable"] is False
        assert math.isnan(entry["l8_final_linf"])


# ---------------------------------------------------------------------------
# TestBaselineStaged remaining 47.5c contract (omitted; persisted-alongside)
# ---------------------------------------------------------------------------
#
# The 47.5b.2 Done note records that ``test_runs_when_flag_set`` (originally
# enumerated under 47.5c) is already covered by
# ``TestBOCLI::test_baseline_staged_invokes_run_staged_optimize`` (and its
# HF=L7 sibling); per that note we do NOT re-add it here.  The two
# remaining 47.5c contracts are pinned below.


class TestBaselineStagedOmitted:
    """Plan 47.5c: ``--baseline`` not passed ⇒ no baseline ever recorded."""

    def test_omitted_when_flag_unset(self, monkeypatch, capsys):
        """Without ``--baseline``: ``run_staged_optimize`` is never called,
        ``result.extras["baseline"]`` is absent, and the side-by-side
        comparison block does not appear in the summary print.
        """
        from dataclasses import replace

        captured: dict[str, BOResult | None] = {"result": None}

        def _fake_run_mfbo(**kwargs):
            stub = _make_bo_result()
            ret = replace(
                stub,
                fidelity_levels=(1, 3),
                hf_level=3,
                report_fields_by_layer={
                    1: "layer1.boundary_gv_err",
                    3: "layer3.max_stab_eig",
                },
                cost_model={1: 0.076, 3: 0.038},
                n_evals_per_fidelity={1: 5, 3: 5},
                wall_time_per_fidelity={1: 0.4, 3: 0.2},
                bounds=((0.5, 20.0),),
                kernel="tension",
                best_x=np.array([10.0]),
                best_params={"sigma": 10.0},
                best_objective=-1.0e-3,
                extras={"n_sentinel_filtered": 0},
            )
            captured["result"] = ret
            return ret

        def _unexpected(**kwargs):
            raise AssertionError(
                "run_staged_optimize must not be invoked without --baseline staged"
            )

        monkeypatch.setattr(bo_cli, "run_mfbo", _fake_run_mfbo)
        monkeypatch.setattr(bo_cli, "run_staged_optimize", _unexpected)

        rc = bo_cli.main(
            [
                "--scheme", "E4",
                "--kernel", "tension",
                "--objective", "layer3.max_stab_eig",
                "--cheap-fidelities", "1",
                "--bounds", "0.5", "20",
                "--budget-evals", "10",
                "--seed", "1",
            ]
        )
        assert rc == 0
        assert captured["result"] is not None
        # No "baseline" key landed under extras.
        assert "baseline" not in captured["result"].extras
        out = capsys.readouterr().out
        # Side-by-side comparison only renders when a baseline is present;
        # without it that block must be silent.
        assert "comparison (side-by-side)" not in out
        # And the standalone "baseline (staged):" header from _print_summary
        # must not appear either.
        assert "baseline (staged):" not in out


class TestBaselineStagedPersisted:
    """Plan 47.5c: ``--baseline staged --persist`` ⇒ both winners on disk."""

    def test_persisted_alongside_bo_result(
        self, monkeypatch, capsys, tmp_path
    ):
        """The persisted JSON contains both ``best_x`` (MF-BO winner) and
        ``extras.baseline.best_x`` (staged winner), plus method tags."""
        from dataclasses import replace

        from stencil_gen.optimizer import OptimizeResult as _OR

        def _fake_run_mfbo(**kwargs):
            stub = _make_bo_result()
            return replace(
                stub,
                fidelity_levels=(1, 3),
                hf_level=3,
                report_fields_by_layer={
                    1: "layer1.boundary_gv_err",
                    3: "layer3.max_stab_eig",
                },
                cost_model={1: 0.076, 3: 0.038},
                n_evals_per_fidelity={1: 5, 3: 5},
                wall_time_per_fidelity={1: 0.4, 3: 0.2},
                bounds=((0.5, 20.0),),
                kernel="tension",
                best_x=np.array([10.0]),
                best_params={"sigma": 10.0},
                best_objective=-1.0e-3,
                extras={"n_sentinel_filtered": 0},
            )

        def _fake_run_staged_optimize(**kwargs):
            return _OR(
                best_params={"sigma": 11.0},
                best_x=np.array([11.0]),
                best_objective=-2.0e-3,
                best_report={"failed_layer": None},
                method="staged",
                converged=True,
                n_evals=42,
                compute_time=1.5,
                history=[],
                extras={
                    "stage": "validated",
                    "validator_ranking": [
                        (np.array([11.0]), -2.0e-3),
                        (np.array([12.0]), -1.5e-3),
                    ],
                },
            )

        # Redirect persist target to ``tmp_path`` (default-binding caveat:
        # wrap save_bo_run instead of patching BO_RUNS_DIR — see 47.5b.2).
        import sweeps._bo_io as bo_io_mod

        original_save = bo_io_mod.save_bo_run

        def _save_to_tmp(result, directory=tmp_path):
            return original_save(result, directory=directory)

        monkeypatch.setattr(bo_cli, "run_mfbo", _fake_run_mfbo)
        monkeypatch.setattr(bo_cli, "run_staged_optimize", _fake_run_staged_optimize)
        monkeypatch.setattr(bo_io_mod, "save_bo_run", _save_to_tmp)

        rc = bo_cli.main(
            [
                "--scheme", "E4",
                "--kernel", "tension",
                "--objective", "layer3.max_stab_eig",
                "--cheap-fidelities", "1",
                "--bounds", "0.5", "20",
                "--budget-evals", "10",
                "--seed", "1",
                "--baseline", "staged",
                "--persist",
            ]
        )
        assert rc == 0

        persisted = tmp_path / "E4_tension_layer3_max_stab_eig_1.json"
        assert persisted.exists(), (
            f"expected persisted JSON at {persisted}; "
            f"tmp_path contents: {list(tmp_path.iterdir())}"
        )
        payload = json.loads(persisted.read_text())

        # MF-BO winner top-level.
        assert payload["best_x"] == [10.0]
        assert payload["best_objective"] == pytest.approx(-1.0e-3)
        assert payload["method"] == "BoTorch-qMFKG"

        # Staged winner under extras.baseline.
        baseline = payload["extras"]["baseline"]
        assert baseline["best_x"] == [11.0]
        assert baseline["best_objective"] == pytest.approx(-2.0e-3)
        assert baseline["method"] == "staged"
        # n_evals_at_hf was derived from validator_ranking length.
        assert baseline["n_evals_at_hf"] == 2
        # No failure path was taken.
        assert "error" not in baseline
