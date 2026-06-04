"""Shared helpers for sweep scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, is_dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

KNOWN_VALUES_PATH = Path(__file__).parent / "known_values.json"

# Stability threshold: eigenvalues below this are considered numerically zero
# (i.e., the operator is stable).
STABILITY_TOL = 1e-10

# Scheme parameters: (p, q, nextra, nu) for each named scheme.
SCHEME_PARAMS = {
    "E2": {"p": 1, "q": 1, "nextra": 1, "nu": 1, "label": "E2_1"},
    "E4": {"p": 2, "q": 3, "nextra": 0, "nu": 1, "label": "E4_1"},
}


@dataclass
class SweepResult:
    """Result of a single parameter sweep point."""

    parameter: float
    eigenvalue: float
    stable: bool
    n: int | None = None
    label: str = ""
    extra: dict = field(default_factory=dict)


def print_table(
    title: str,
    headers: list[str],
    rows: list[list[str]],
    *,
    col_widths: list[int] | None = None,
) -> None:
    """Print a formatted table to stdout."""
    if col_widths is None:
        col_widths = [
            max(len(h), max((len(str(r)) for r in col), default=0)) + 2
            for h, col in zip(headers, zip(*rows))
        ]
        # Ensure header widths are respected
        col_widths = [max(w, len(h) + 2) for w, h in zip(col_widths, headers)]

    print(f"\n{'=' * sum(col_widths)}")
    print(f"  {title}")
    print(f"{'=' * sum(col_widths)}")
    header_line = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * sum(col_widths))
    for row in rows:
        print("".join(str(v).ljust(w) for v, w in zip(row, col_widths)))
    print()


class _KnownValuesEncoder(json.JSONEncoder):
    """Encoder for ``known_values.json`` that handles types reachable via
    ``_report_to_dict`` (complex, numpy scalars/arrays, dataclasses, Path).

    Mirrors :class:`sweeps._pareto_io._ParetoEncoder` deliberately rather than
    importing it: ``_common.py`` is a no-pareto/no-pymoo module and the
    encoder is small enough to duplicate. ``complex`` is encoded as
    ``[real, imag]`` to match the Pareto-front convention; the dict-form
    encoding used by ``brady2d_cli.py`` writes to a separate calibration JSON
    and does not feed ``known_values.json``.
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        if is_dataclass(o) and not isinstance(o, type):
            return asdict(o)
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, complex):
            return [o.real, o.imag]
        return super().default(o)


def load_known_values() -> dict:
    """Load known optimal values from JSON."""
    if not KNOWN_VALUES_PATH.exists():
        return {}
    with open(KNOWN_VALUES_PATH) as f:
        return json.load(f)


def save_known_values(data: dict) -> None:
    """Save known optimal values to JSON."""
    with open(KNOWN_VALUES_PATH, "w") as f:
        json.dump(data, f, indent=2, cls=_KnownValuesEncoder)
        f.write("\n")


def print_sweep_table(
    label: str,
    results: dict[int, list[tuple[float, float]]],
    *,
    param_label: str = "param",
    gv_by_param: dict[float, float] | None = None,
) -> None:
    """Print formatted sweep table with stability classification.

    If ``gv_by_param`` is given, append a ``gv_err`` column whose value is
    looked up by the row's parameter (GV is independent of n, so the same
    value repeats across grid sizes).
    """
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")
    for n, rows in sorted(results.items()):
        print(f"\n  n = {n}")
        if gv_by_param is None:
            print(f"  {param_label:>10s}  {'stab_eig':>14s}  {'status':>10s}")
            print(f"  {'-'*10}  {'-'*14}  {'-'*10}")
            for val, se in rows:
                status = "STABLE" if se < STABILITY_TOL else "unstable"
                print(f"  {val:10.4f}  {se:14.6e}  {status:>10s}")
        else:
            print(
                f"  {param_label:>10s}  {'stab_eig':>14s}  "
                f"{'status':>10s}  {'gv_err':>14s}"
            )
            print(f"  {'-'*10}  {'-'*14}  {'-'*10}  {'-'*14}")
            for val, se in rows:
                status = "STABLE" if se < STABILITY_TOL else "unstable"
                gv = gv_by_param.get(float(val), float("nan"))
                print(
                    f"  {val:10.4f}  {se:14.6e}  {status:>10s}  {gv:14.6e}"
                )

    # Summary: best value per n
    print(f"\n  --- Best {param_label} (min stability eigenvalue) ---")
    for n, rows in sorted(results.items()):
        best = min(rows, key=lambda r: r[1])
        stable = "STABLE" if best[1] < STABILITY_TOL else "unstable"
        print(f"  n={n:3d}: {param_label}={best[0]:.4f}, stab_eig={best[1]:.6e} [{stable}]")


def report_stable_ranges(
    results: dict[int, list[tuple[float, float]]],
    *,
    param_label: str = "param",
) -> None:
    """Print stable parameter ranges and counts per grid size."""
    for n, rows in sorted(results.items()):
        stable_vals = [v for v, se in rows if se < STABILITY_TOL]
        n_stable = len(stable_vals)
        if stable_vals:
            print(f"  n={n}: {n_stable}/{len(rows)} stable, "
                  f"{param_label} range [{min(stable_vals):.4f}, {max(stable_vals):.4f}]")
        else:
            best = min(rows, key=lambda r: r[1])
            print(f"  n={n}: no stable {param_label} found, "
                  f"best stab_eig={best[1]:.6e} at {param_label}={best[0]:.4f}")


def bisect_threshold(f, a, b, threshold, *, tol=1e-4, maxiter=60):
    """Bisect to find x where f(x) crosses threshold from above.

    Assumes f(a) > threshold and f(b) < threshold.
    Returns x such that f(x) is near threshold.
    """
    for _ in range(maxiter):
        mid = (a + b) / 2
        if (b - a) < tol:
            break
        if f(mid) > threshold:
            a = mid
        else:
            b = mid
    return (a + b) / 2
