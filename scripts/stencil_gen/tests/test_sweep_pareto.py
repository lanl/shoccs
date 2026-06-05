"""Unit tests for :mod:`sweeps.pareto` (plan 45.3c / 45.4c).

Covers:

- :func:`sweeps.pareto._mangle_objectives` — filename mangling round-trip.
- :func:`sweeps.pareto.main` argparse surface — minimal accepting invocation,
  single-objective rejection, odd-bound-parity rejection.
- ``python -m sweeps pareto --help`` subprocess smoke — confirms the dispatch
  wiring from ``sweeps/__main__.py`` survived the registration of the new
  subcommand.
- :mod:`sweeps._pareto_io` — per-run JSON persistence save/load/iter
  (plan 45.4a).
- ``--persist`` wiring in :func:`sweeps.pareto.main` — verifies the CLI
  actually calls :func:`save_pareto_front` when requested and skips it
  otherwise (plan 45.4c review follow-up to 45.4b).

``TestParetoCLI::test_argparse_accepts_minimal_invocation`` stubs
:func:`sweeps.pareto.run_nsga2` via ``monkeypatch`` so no pymoo / brady2d
pipeline is entered — keeps the test in the fast suite.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from stencil_gen.pareto import ParetoPoint, ParetoResult

from sweeps import pareto as pareto_cli
from sweeps._pareto_io import (
    iter_pareto_fronts,
    load_pareto_front,
    save_pareto_front,
)


# ---------------------------------------------------------------------------
# _mangle_objectives
# ---------------------------------------------------------------------------


class TestMangleObjectives:
    def test_roundtrip_legible(self):
        fields = ["layer1.boundary_gv_err", "layer_bl42.max_spectral_abscissa"]
        assert (
            pareto_cli._mangle_objectives(fields)
            == "layer1_boundary_gv_err__layer_bl42_max_spectral_abscissa"
        )

    def test_order_preserved(self):
        a = ["layer1.boundary_gv_err", "layer3.max_stab_eig"]
        b = ["layer3.max_stab_eig", "layer1.boundary_gv_err"]
        assert pareto_cli._mangle_objectives(a) != pareto_cli._mangle_objectives(b)


# ---------------------------------------------------------------------------
# main(argv) argparse / dispatch
# ---------------------------------------------------------------------------


def _stub_result(
    *,
    objective_fields: tuple[str, ...],
    scheme: str = "E4",
    kernel: str = "classical",
    bounds: tuple[tuple[float, float], ...] = ((-2.0, 2.0), (0.05, 2.0)),
    pop_size: int = 6,
    n_gen: int = 2,
    seed: int = 1,
) -> ParetoResult:
    """Build a minimal, deterministic ParetoResult for CLI stubbing."""
    n_obj = len(objective_fields)
    n_var = len(bounds)
    pt = ParetoPoint(
        x=np.zeros(n_var, dtype=float),
        params={"alpha": [0.0] * n_var} if kernel == "classical" else {},
        objectives=np.arange(n_obj, dtype=float) + 1.0,
        report={},
    )
    return ParetoResult(
        front=(pt,),
        objective_fields=objective_fields,
        scheme=scheme,
        kernel=kernel,
        bounds=bounds,
        method="NSGA-II",
        pop_size=pop_size,
        n_gen=n_gen,
        n_evals=pop_size * n_gen,
        seed=seed,
        compute_time=0.001,
        hv_trace=(0.1, 0.2),
        ref_point=tuple(1.0 for _ in range(n_obj)),
        extras={"n_sentinel_filtered": 0, "hv_n_nds": (1, 1)},
    )


class TestParetoCLI:
    def test_argparse_accepts_minimal_invocation(self, monkeypatch, capsys):
        """Minimal classical/E4 invocation with mocked run_nsga2 exits 0."""
        calls: list[dict] = []

        def _fake_run_nsga2(**kwargs):
            calls.append(kwargs)
            return _stub_result(
                objective_fields=tuple(kwargs["report_fields"]),
                scheme=kwargs["scheme"],
                kernel=kwargs["kernel"],
                bounds=tuple(kwargs["bounds"]),
                pop_size=kwargs["pop_size"],
                n_gen=kwargs["n_gen"],
                seed=kwargs["seed"],
            )

        monkeypatch.setattr(pareto_cli, "run_nsga2", _fake_run_nsga2)

        rc = pareto_cli.main(
            [
                "--scheme", "E4",
                "--kernel", "classical",
                "--objectives",
                "layer1.boundary_gv_err",
                "layer3.max_stab_eig",
                "--pop-size", "6",
                "--n-gen", "2",
                "--seed", "1",
            ]
        )
        assert rc == 0
        assert len(calls) == 1
        call = calls[0]
        assert call["scheme"] == "E4"
        assert call["kernel"] == "classical"
        assert list(call["report_fields"]) == [
            "layer1.boundary_gv_err",
            "layer3.max_stab_eig",
        ]
        assert call["pop_size"] == 6
        assert call["n_gen"] == 2
        assert call["seed"] == 1
        # Default bounds for (E4, classical) are 2D, matching the classical kernel.
        assert len(call["bounds"]) == 2
        out = capsys.readouterr().out
        assert "NSGA-II" in out
        assert "layer1.boundary_gv_err" in out

    def test_argparse_rejects_single_objective(self):
        """A single --objectives value exits non-zero (min 2 enforced)."""
        with pytest.raises(SystemExit) as exc_info:
            pareto_cli.main(
                [
                    "--scheme", "E4",
                    "--kernel", "classical",
                    "--objectives", "layer3.max_stab_eig",
                    "--pop-size", "6",
                    "--n-gen", "2",
                ]
            )
        assert exc_info.value.code != 0

    def test_argparse_rejects_bad_bounds_parity(self):
        """An odd number of --bounds values exits non-zero (pairs required)."""
        with pytest.raises(SystemExit) as exc_info:
            pareto_cli.main(
                [
                    "--scheme", "E4",
                    "--kernel", "classical",
                    "--objectives",
                    "layer1.boundary_gv_err",
                    "layer3.max_stab_eig",
                    "--bounds", "-2.0", "2.0", "0.05",
                    "--pop-size", "6",
                    "--n-gen", "2",
                ]
            )
        assert exc_info.value.code != 0

    def test_dispatch_registered(self):
        """`python -m sweeps pareto --help` is routed through __main__."""
        stencil_gen_dir = Path(__file__).resolve().parent.parent
        env = os.environ.copy()
        env.setdefault("SYMPY_CACHE_SIZE", "50000")
        proc = subprocess.run(
            [sys.executable, "-m", "sweeps", "pareto", "--help"],
            cwd=str(stencil_gen_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, (
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
        # The top-level dispatch parser prints its own --help for the
        # subparser; spot-check a handful of options.
        for needle in ("--objectives", "--pop-size", "--n-gen", "--seed"):
            assert needle in proc.stdout, (
                f"missing {needle!r} in `python -m sweeps pareto --help` output"
            )

    def test_persist_flag_invokes_save_pareto_front(self, monkeypatch, tmp_path, capsys):
        """``--persist`` must call :func:`save_pareto_front` exactly once.

        Stubs both :func:`run_nsga2` (to avoid pymoo/brady2d) and
        :func:`save_pareto_front` (to keep the working-copy
        ``sweeps/pareto_fronts/`` untouched — the CLI passes no ``directory=``
        override, so an unmocked save would write into the repo).
        """
        stub = _stub_result(
            objective_fields=("layer1.boundary_gv_err", "layer3.max_stab_eig"),
        )
        monkeypatch.setattr(pareto_cli, "run_nsga2", lambda **_: stub)

        recorder: dict = {}
        sentinel = tmp_path / "fake.json"

        def _fake_save(result, *args, **kwargs):
            recorder["called"] = recorder.get("called", 0) + 1
            recorder["result"] = result
            return sentinel

        monkeypatch.setattr(pareto_cli, "save_pareto_front", _fake_save)

        rc = pareto_cli.main(
            [
                "--scheme", "E4",
                "--kernel", "classical",
                "--objectives",
                "layer1.boundary_gv_err",
                "layer3.max_stab_eig",
                "--pop-size", "6",
                "--n-gen", "2",
                "--seed", "1",
                "--persist",
            ]
        )
        assert rc == 0
        assert recorder.get("called") == 1
        assert recorder["result"] is stub
        out = capsys.readouterr().out
        assert "persisted front to" in out
        assert str(sentinel) in out

    def test_no_persist_does_not_invoke_save_pareto_front(
        self, monkeypatch, capsys
    ):
        """Without ``--persist`` the IO helper must not be called."""
        stub = _stub_result(
            objective_fields=("layer1.boundary_gv_err", "layer3.max_stab_eig"),
        )
        monkeypatch.setattr(pareto_cli, "run_nsga2", lambda **_: stub)

        recorder: dict = {}

        def _fake_save(result, *args, **kwargs):
            recorder["called"] = recorder.get("called", 0) + 1
            return Path("/should/not/be/written.json")

        monkeypatch.setattr(pareto_cli, "save_pareto_front", _fake_save)

        rc = pareto_cli.main(
            [
                "--scheme", "E4",
                "--kernel", "classical",
                "--objectives",
                "layer1.boundary_gv_err",
                "layer3.max_stab_eig",
                "--pop-size", "6",
                "--n-gen", "2",
                "--seed", "1",
            ]
        )
        assert rc == 0
        assert "called" not in recorder
        out = capsys.readouterr().out
        assert "persisted front to" not in out

    def test_persist_and_validate_writes_cpp_validation_to_json(
        self, monkeypatch, tmp_path
    ):
        """``--persist --validate-with-cpp`` must capture ``cpp_validation`` on disk.

        Regression for plan 45.5a.1: ``_run_front_cpp_validation`` mutates
        ``result.extras``, so it MUST run before ``save_pareto_front``; otherwise
        the JSON misses the diagnostic payload.  This test monkeypatches the
        validator (to avoid spinning up shoccs) but routes through the real
        :func:`save_pareto_front` (with ``directory=tmp_path``) to catch the
        on-disk shape.
        """
        stub = _stub_result(
            objective_fields=("layer1.boundary_gv_err", "layer3.max_stab_eig"),
        )
        monkeypatch.setattr(pareto_cli, "run_nsga2", lambda **_: stub)

        fake_validation = [
            {
                "x": [0.0, 0.0],
                "params": {"alpha": [0.0, 0.0]},
                "l8_stable": True,
                "l8_final_linf": 1.2e-3,
                "wall_time_s": 0.42,
            }
        ]

        def _fake_validate(result, *args, **kwargs):
            # Assert ordering: persistence must not have fired yet.  If
            # save_pareto_front ran first, extras already holds the JSON-safe
            # snapshot (no in-place cpp_validation from a prior run because
            # the stub starts without one).  We only check the call count.
            return fake_validation

        monkeypatch.setattr(
            pareto_cli, "_run_front_cpp_validation", _fake_validate
        )

        real_save = pareto_cli.save_pareto_front
        save_calls: list[ParetoResult] = []

        def _recording_save(result, *args, **kwargs):
            # Capture the ParetoResult at the moment of persistence so we can
            # confirm validation already mutated ``extras``.
            save_calls.append(result)
            # Delegate to the real implementation but force ``tmp_path`` so
            # the working-copy ``sweeps/pareto_fronts/`` stays untouched.
            return real_save(result, directory=tmp_path)

        monkeypatch.setattr(pareto_cli, "save_pareto_front", _recording_save)

        rc = pareto_cli.main(
            [
                "--scheme", "E4",
                "--kernel", "classical",
                "--objectives",
                "layer1.boundary_gv_err",
                "layer3.max_stab_eig",
                "--pop-size", "6",
                "--n-gen", "2",
                "--seed", "1",
                "--persist",
                "--validate-with-cpp",
            ]
        )
        assert rc == 0
        assert len(save_calls) == 1
        # The ParetoResult at save time already carries the validation payload
        # (ordering fix: validate-before-persist).
        assert save_calls[0].extras.get("cpp_validation") == fake_validation

        written = tmp_path / (
            "E4_classical_layer1_boundary_gv_err__layer3_max_stab_eig.json"
        )
        assert written.exists()
        payload = json.loads(written.read_text())
        assert payload["extras"].get("cpp_validation") == fake_validation

    def test_persist_without_validate_omits_cpp_validation(
        self, monkeypatch, tmp_path
    ):
        """``--persist`` alone must not emit a ``cpp_validation`` entry."""
        stub = _stub_result(
            objective_fields=("layer1.boundary_gv_err", "layer3.max_stab_eig"),
        )
        monkeypatch.setattr(pareto_cli, "run_nsga2", lambda **_: stub)

        # If this fires, the CLI is inadvertently validating without the flag.
        def _unexpected_validate(*args, **kwargs):
            raise AssertionError(
                "_run_front_cpp_validation called without --validate-with-cpp"
            )

        monkeypatch.setattr(
            pareto_cli, "_run_front_cpp_validation", _unexpected_validate
        )

        real_save = pareto_cli.save_pareto_front
        monkeypatch.setattr(
            pareto_cli,
            "save_pareto_front",
            lambda result, *a, **kw: real_save(result, directory=tmp_path),
        )

        rc = pareto_cli.main(
            [
                "--scheme", "E4",
                "--kernel", "classical",
                "--objectives",
                "layer1.boundary_gv_err",
                "layer3.max_stab_eig",
                "--pop-size", "6",
                "--n-gen", "2",
                "--seed", "1",
                "--persist",
            ]
        )
        assert rc == 0
        written = tmp_path / (
            "E4_classical_layer1_boundary_gv_err__layer3_max_stab_eig.json"
        )
        assert written.exists()
        payload = json.loads(written.read_text())
        assert "cpp_validation" not in payload["extras"]

    def test_validate_without_persist_does_not_write_file(
        self, monkeypatch, tmp_path
    ):
        """``--validate-with-cpp`` alone must not write JSON to disk."""
        stub = _stub_result(
            objective_fields=("layer1.boundary_gv_err", "layer3.max_stab_eig"),
        )
        monkeypatch.setattr(pareto_cli, "run_nsga2", lambda **_: stub)
        monkeypatch.setattr(
            pareto_cli,
            "_run_front_cpp_validation",
            lambda result, *a, **kw: [
                {
                    "x": [0.0, 0.0],
                    "params": {"alpha": [0.0, 0.0]},
                    "l8_stable": True,
                    "l8_final_linf": 1.0e-3,
                    "wall_time_s": 0.1,
                }
            ],
        )

        def _unexpected_save(*args, **kwargs):
            raise AssertionError(
                "save_pareto_front called without --persist"
            )

        monkeypatch.setattr(pareto_cli, "save_pareto_front", _unexpected_save)

        rc = pareto_cli.main(
            [
                "--scheme", "E4",
                "--kernel", "classical",
                "--objectives",
                "layer1.boundary_gv_err",
                "layer3.max_stab_eig",
                "--pop-size", "6",
                "--n-gen", "2",
                "--seed", "1",
                "--validate-with-cpp",
            ]
        )
        assert rc == 0
        # The validator should have written into the (in-memory) ParetoResult.
        assert stub.extras.get("cpp_validation") is not None
        # No file should have been written into tmp_path.
        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# _run_front_cpp_validation (plan 45.5b)
# ---------------------------------------------------------------------------


class TestValidateWithCpp:
    """Covers :func:`sweeps.pareto._run_front_cpp_validation` (plan 45.5b).

    Exercises the four observable contracts:

    1. Global skip when ``result.kernel`` is not in ``_CPP_SUPPORTED_KERNELS``
       (returns ``None``, never enters the per-member loop).
    2. Member count is capped at ``_CPP_VALIDATION_MAX_MEMBERS`` (10), in
       ascending-``objectives[0]`` order, for fronts larger than the cap.
    3. Per-member entry shape on success: ``x``, ``params``, ``l8_stable``,
       ``l8_final_linf``, ``wall_time_s`` (+ ``cpp_cutcell_violates_197_288``
       when the scheme/kernel pair is E4-classical).
    4. A raising :func:`brady2d_stability_score` records ``l8_error`` on the
       failing entry and lets the remaining members validate — matches the
       ``sweeps.optimize._run_cpp_validation`` contract.
    """

    @staticmethod
    def _fake_binary(tmp_path, monkeypatch):
        """Make ``pareto_cli.SHOCCS_BINARY`` appear to exist for the duration."""
        fake = tmp_path / "shoccs"
        fake.write_text("")
        monkeypatch.setattr(pareto_cli, "SHOCCS_BINARY", fake)

    @staticmethod
    def _make_result_n_members(
        n: int,
        *,
        scheme: str = "E4",
        kernel: str = "classical",
    ) -> ParetoResult:
        """Build an N-member ParetoResult with ``objectives[0]`` ascending by index."""
        front = tuple(
            _stub_point(
                x=[-0.5 + 0.01 * i, 0.2],
                objectives=[0.01 * i, 0.5 - 0.01 * i],
            )
            for i in range(n)
        )
        return ParetoResult(
            front=front,
            objective_fields=(
                "layer1.boundary_gv_err",
                "layer_bl42.max_spectral_abscissa",
            ),
            scheme=scheme,
            kernel=kernel,
            bounds=((-2.0, 2.0), (0.05, 2.0)),
            method="NSGA-II",
            pop_size=max(n, 1),
            n_gen=1,
            n_evals=max(n, 1),
            seed=1,
            compute_time=0.01,
            hv_trace=(0.1,),
            ref_point=(1.0, 1.0),
            extras={},
        )

    def test_skips_on_unsupported_kernel(self, tmp_path, monkeypatch, capsys):
        """Kernel absent from ``_CPP_SUPPORTED_KERNELS`` → ``None`` + 'skipped'."""
        # The binary is wired up so the skip path is unambiguously the
        # kernel-not-supported branch, not the missing-binary branch.
        self._fake_binary(tmp_path, monkeypatch)

        def _unexpected_call(*args, **kwargs):
            raise AssertionError(
                "brady2d_stability_score called for unsupported kernel"
            )

        monkeypatch.setattr(
            pareto_cli, "brady2d_stability_score", _unexpected_call
        )

        result = self._make_result_n_members(3, kernel="phs")
        out = pareto_cli._run_front_cpp_validation(result)
        assert out is None
        captured = capsys.readouterr().out
        assert "skipped" in captured
        assert "phs" in captured

    def test_caps_at_10_members(self, tmp_path, monkeypatch):
        """A 25-member front validates only 10 (ascending by objectives[0])."""
        from stencil_gen.brady2d_stability import StabilityReport

        self._fake_binary(tmp_path, monkeypatch)
        monkeypatch.setattr(
            pareto_cli,
            "brady2d_stability_score",
            lambda *a, **kw: StabilityReport(
                layer8={
                    "stable": True,
                    "final_linf": 1e-4,
                    "wall_time_s": 0.1,
                },
            ),
        )

        result = self._make_result_n_members(25)
        out = pareto_cli._run_front_cpp_validation(result)
        assert out is not None
        assert len(out) == 10
        # Deterministic: the first 10 have the 10 smallest objectives[0] values,
        # i.e. indices 0..9 — their x[0] = -0.5 + 0.01 * i.
        xs0 = [entry["x"][0] for entry in out]
        assert xs0 == pytest.approx([-0.5 + 0.01 * i for i in range(10)])

    def test_records_per_member_shape(self, tmp_path, monkeypatch):
        """Every successful entry exposes the documented key set."""
        from stencil_gen.brady2d_stability import StabilityReport

        self._fake_binary(tmp_path, monkeypatch)
        monkeypatch.setattr(
            pareto_cli,
            "brady2d_stability_score",
            lambda *a, **kw: StabilityReport(
                layer8={
                    "stable": True,
                    "final_linf": 2.5e-4,
                    "wall_time_s": 0.33,
                },
            ),
        )

        result = self._make_result_n_members(3)
        out = pareto_cli._run_front_cpp_validation(result)
        assert out is not None
        assert len(out) == 3
        for entry in out:
            assert set(entry.keys()) >= {
                "x",
                "params",
                "l8_stable",
                "l8_final_linf",
                "wall_time_s",
            }
            assert entry["l8_stable"] is True
            assert entry["l8_final_linf"] == pytest.approx(2.5e-4)
            assert entry["wall_time_s"] == pytest.approx(0.33)
            # E4-classical with 2D x → the cut-cell 197/288 diagnostic fires.
            assert "cpp_cutcell_violates_197_288" in entry
            assert "l8_error" not in entry

    def test_failure_records_error_not_raises(self, tmp_path, monkeypatch, capsys):
        """A single raising call records ``l8_error``; loop continues."""
        from stencil_gen.brady2d_stability import StabilityReport

        self._fake_binary(tmp_path, monkeypatch)
        call_count = {"n": 0}

        def _maybe_raise(*a, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("synthetic L8 failure")
            return StabilityReport(
                layer8={
                    "stable": True,
                    "final_linf": 1e-4,
                    "wall_time_s": 0.1,
                },
            )

        monkeypatch.setattr(
            pareto_cli, "brady2d_stability_score", _maybe_raise
        )

        result = self._make_result_n_members(3)
        out = pareto_cli._run_front_cpp_validation(result)
        assert out is not None
        assert len(out) == 3

        # First member failed; subsequent members must still validate cleanly.
        assert "l8_error" in out[0]
        assert "synthetic L8 failure" in out[0]["l8_error"]
        assert out[0]["l8_stable"] is False
        assert np.isnan(out[0]["l8_final_linf"])
        assert out[0]["wall_time_s"] == 0.0

        assert "l8_error" not in out[1]
        assert out[1]["l8_stable"] is True
        assert "l8_error" not in out[2]
        assert out[2]["l8_stable"] is True


# ---------------------------------------------------------------------------
# _pareto_io: save / load / iter round-trip
# ---------------------------------------------------------------------------


def _stub_point(
    *,
    x: list[float],
    objectives: list[float],
    params: dict | None = None,
    report: dict | None = None,
) -> ParetoPoint:
    return ParetoPoint(
        x=np.asarray(x, dtype=float),
        params=params if params is not None else {"alpha": list(x)},
        objectives=np.asarray(objectives, dtype=float),
        report=report or {},
    )


def _stub_result_multi_member(
    *,
    scheme: str = "E4",
    kernel: str = "classical",
    objective_fields: tuple[str, ...] = (
        "layer1.boundary_gv_err",
        "layer_bl42.max_spectral_abscissa",
    ),
) -> ParetoResult:
    """Build a 3-member ParetoResult exercising all serialised fields."""
    front = (
        _stub_point(x=[-0.8, 0.16], objectives=[3.6e-2, 0.95]),
        _stub_point(x=[-0.5, 0.20], objectives=[8.0e-2, 0.30]),
        _stub_point(x=[-0.2, 0.25], objectives=[1.0e-1, 1.0e-13]),
    )
    return ParetoResult(
        front=front,
        objective_fields=objective_fields,
        scheme=scheme,
        kernel=kernel,
        bounds=((-2.0, 2.0), (0.05, 2.0)),
        method="NSGA-II",
        pop_size=12,
        n_gen=4,
        n_evals=48,
        seed=1,
        compute_time=1.234,
        hv_trace=(0.1, 0.2, 0.25, 0.3),
        ref_point=(1.1, 1.1),
        extras={"n_sentinel_filtered": 2, "hv_n_nds": (3, 3, 3, 3)},
    )


class TestParetoIO:
    def test_save_creates_file_at_mangled_path(self, tmp_path):
        """Filename must be ``{scheme}_{kernel}_{mangled}.json``."""
        result = _stub_result_multi_member()
        written = save_pareto_front(result, directory=tmp_path)
        expected = tmp_path / (
            "E4_classical_"
            "layer1_boundary_gv_err__layer_bl42_max_spectral_abscissa.json"
        )
        assert written == expected
        assert expected.exists()
        assert expected.stat().st_size > 0

    def test_roundtrip_preserves_objectives(self, tmp_path):
        """``load_pareto_front`` should round-trip every front member's arrays."""
        result = _stub_result_multi_member()
        written = save_pareto_front(result, directory=tmp_path)
        loaded = load_pareto_front(written)

        assert loaded["scheme"] == result.scheme
        assert loaded["kernel"] == result.kernel
        assert loaded["method"] == result.method
        assert tuple(loaded["objective_fields"]) == result.objective_fields
        assert loaded["pop_size"] == result.pop_size
        assert loaded["n_gen"] == result.n_gen
        assert loaded["n_evals"] == result.n_evals
        assert loaded["seed"] == result.seed
        assert loaded["compute_time"] == result.compute_time
        assert tuple(loaded["ref_point"]) == result.ref_point
        assert tuple(loaded["hv_trace"]) == result.hv_trace

        assert len(loaded["front"]) == len(result.front)
        for loaded_pt, pt in zip(loaded["front"], result.front):
            assert loaded_pt["x"] == pt.x.tolist()
            assert loaded_pt["objectives"] == pt.objectives.tolist()

    def test_serializer_handles_numpy_arrays(self, tmp_path):
        """numpy ndarrays and scalars in ``extras`` must not raise TypeError."""
        result = _stub_result_multi_member()
        # Swap in a numpy array and a numpy scalar via dataclass replace-equivalent.
        # ``extras`` is a plain dict on a frozen dataclass, so mutate in place.
        result.extras["np_array"] = np.array([1.0, 2.5, 3.75])
        result.extras["np_scalar"] = np.float64(42.0)
        written = save_pareto_front(result, directory=tmp_path)
        raw = json.loads(written.read_text())
        assert raw["extras"]["np_array"] == [1.0, 2.5, 3.75]
        assert raw["extras"]["np_scalar"] == 42.0

    def test_iter_discovers_multiple_files(self, tmp_path):
        """``iter_pareto_fronts`` yields every ``*.json`` in the directory."""
        a = save_pareto_front(
            _stub_result_multi_member(
                objective_fields=(
                    "layer1.boundary_gv_err",
                    "layer3.max_stab_eig",
                )
            ),
            directory=tmp_path,
        )
        b = save_pareto_front(
            _stub_result_multi_member(
                scheme="E2",
                objective_fields=(
                    "layer1.boundary_gv_err",
                    "layer_bl42.max_spectral_abscissa",
                ),
            ),
            directory=tmp_path,
        )
        discovered = list(iter_pareto_fronts(tmp_path))
        assert set(discovered) == {a, b}

    def test_iter_on_missing_directory_yields_nothing(self, tmp_path):
        """A nonexistent directory is a no-op, not an error."""
        missing = tmp_path / "does_not_exist"
        assert list(iter_pareto_fronts(missing)) == []
