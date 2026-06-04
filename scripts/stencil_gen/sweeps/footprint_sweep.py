"""Stencil footprint (nextra) sweep for E4 tension kernel.

Extracted from TestCorrectedFootprint, TestFootprintE4Quick,
TestFootprintSweep, TestFootprintPenalty in test_phs.py.

Sweeps nextra (number of extra boundary rows) across sigma values
to determine how stencil footprint affects stability.  Optionally
includes a conservation penalty (gamma) dimension.

Three phases:
  1. nextra x sigma sweep — stability landscape per nextra
  2. nextra x sigma x gamma penalty sweep — stability-conservation trade-off
  3. Grid independence — verify best parameters across grid sizes

Usage:
    uv run python -m sweeps.footprint_sweep
    uv run python -m sweeps.footprint_sweep --n-sigma 10  # quick smoke test
    uv run python -m sweeps.footprint_sweep --update-known-values
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from stencil_gen.phs import (
    build_diff_matrix_rbf_penalty,
    stability_eigenvalue,
    stability_eigenvalue_from_matrix,
)

from ._common import STABILITY_TOL, load_known_values, save_known_values
from .gv_objectives import boundary_gv_error_max

# E4 scheme parameters (footprint sweep is E4-only)
P = 2
Q = 3
NU = 1


def run_nextra_sigma_sweep(
    n: int,
    nextra_values: list[int],
    n_sigma: int,
    sigma_max: float,
    *,
    include_gv: bool = False,
) -> tuple[dict[int, dict], dict[int, dict[float, float]] | None]:
    """Phase 1: nextra x sigma sweep.

    For each nextra, sweeps sigma in [0, sigma_max] and reports stability.
    Returns ``(results, gv_by_nx_sigma)`` where ``results`` maps
    ``nextra -> {best_sigma, best_se, n_stable, total, rows}`` and
    ``gv_by_nx_sigma`` maps ``nextra -> {sigma -> boundary_gv_error_max}``
    (grid-size independent) when ``include_gv`` is set, else ``None``.
    """
    sigmas = np.concatenate([
        [0.0],
        np.logspace(np.log10(0.01), np.log10(sigma_max), n_sigma),
    ])

    results = {}
    gv_by_nx_sigma: dict[int, dict[float, float]] | None = (
        {} if include_gv else None
    )
    for nx in nextra_values:
        r = Q + 1 + nx
        if n < 2 * r:
            print(f"  nextra={nx}: grid too small (n={n} < 2*r={2*r}), skipping")
            continue

        rows = []
        for sigma in sigmas:
            se = stability_eigenvalue(
                n, p=P, q=Q, epsilon=sigma,
                kernel="tension", nu=NU, nextra=nx,
            )
            rows.append((float(sigma), se))

        best_sigma, best_se = min(rows, key=lambda r: r[1])
        n_stable = sum(1 for _, se in rows if se < STABILITY_TOL)

        results[nx] = {
            "best_sigma": best_sigma,
            "best_se": best_se,
            "n_stable": n_stable,
            "total": len(rows),
            "rows": rows,
        }

        if include_gv:
            gv_for_nx: dict[float, float] = {}
            for sigma in sigmas:
                gv_for_nx[float(sigma)] = boundary_gv_error_max(
                    p=P, q=Q, nextra=nx, nu=NU,
                    sigma=float(sigma), kernel="tension",
                )
            gv_by_nx_sigma[nx] = gv_for_nx

    # Print table
    print(f"\n{'='*80}")
    print(f"  E4 Tension — nextra x sigma Sweep (n={n})")
    print(f"{'='*80}")

    # Header
    nx_col_width = 32 if include_gv else 16
    header = f"  {'sigma':>10s}"
    for nx in nextra_values:
        if nx in results:
            header += f"  {'nx=' + str(nx):>{nx_col_width}s}"
    print(header)
    if include_gv:
        subheader = f"  {'':>10s}"
        for nx in nextra_values:
            if nx in results:
                subheader += f"  {'stab_eig':>16s}{'gv_err':>16s}"
        print(subheader)
    divider = f"  {'-'*10}"
    for nx in nextra_values:
        if nx in results:
            divider += f"  {'-'*nx_col_width}"
    print(divider)

    # Print every 5th row to keep output readable
    n_rows = len(sigmas)
    for idx in range(0, n_rows, max(1, n_rows // 20)):
        line = f"  {sigmas[idx]:10.4f}"
        for nx in nextra_values:
            if nx in results:
                sigma_at_idx, se = results[nx]["rows"][idx]
                if include_gv:
                    marker = " *" if se < STABILITY_TOL else "  "
                    gv = gv_by_nx_sigma[nx][sigma_at_idx]
                    line += f"  {se:14.6e}{marker}{gv:14.6e}  "
                else:
                    marker = " *" if se < STABILITY_TOL else ""
                    line += f"  {se:14.6e}{marker}"
        print(line)

    # Populate GV-optimal-feasible stats per nextra when include_gv is set.
    # "Feasible" here = stable at the primary grid size n (matching the scope
    # of the row data we have); cross-grid re-checks happen in 40.5c.
    if include_gv:
        for nx in nextra_values:
            if nx not in results:
                continue
            stable_gv = [
                (sigma_at, gv_by_nx_sigma[nx][sigma_at])
                for sigma_at, se in results[nx]["rows"]
                if se < STABILITY_TOL
            ]
            if stable_gv:
                best_gv_sigma, best_gv = min(stable_gv, key=lambda t: t[1])
            else:
                best_gv_sigma, best_gv = None, None
            results[nx]["best_stable_gv_sigma"] = best_gv_sigma
            results[nx]["best_stable_gv"] = best_gv

    # Summary
    print(f"\n  {'='*70}")
    print(f"  Summary: Stability per nextra")
    print(f"  {'='*70}")
    print(f"  {'nextra':>6s}  {'t':>3s}  {'r':>3s}  "
          f"{'extra DOF':>9s}  {'best sigma':>10s}  {'stab_eig':>16s}  "
          f"{'stable':>8s}  {'status':>10s}")
    print(f"  {'-'*6}  {'-'*3}  {'-'*3}  "
          f"{'-'*9}  {'-'*10}  {'-'*16}  "
          f"{'-'*8}  {'-'*10}")

    for nx in nextra_values:
        if nx not in results:
            continue
        res = results[nx]
        t = P + Q + 1 + nx
        r = Q + 1 + nx
        extra_dof = r * (P + nx)
        status = "STABLE" if res["best_se"] < STABILITY_TOL else "unstable"
        print(f"  {nx:6d}  {t:3d}  {r:3d}  "
              f"{extra_dof:9d}  {res['best_sigma']:10.4f}  {res['best_se']:16.6e}  "
              f"{res['n_stable']:>3d}/{res['total']:<3d}  {status:>10s}")

    if include_gv:
        print(f"\n  {'='*70}")
        print(f"  Summary: Best stable (by GV error) per nextra")
        print(f"  {'='*70}")
        print(f"  {'nextra':>6s}  {'gv sigma':>10s}  {'gv_err':>16s}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*16}")
        best_overall: tuple[int, float, float] | None = None
        for nx in nextra_values:
            if nx not in results:
                continue
            gv_sigma = results[nx].get("best_stable_gv_sigma")
            gv_err = results[nx].get("best_stable_gv")
            if gv_sigma is None:
                print(f"  {nx:6d}  {'--':>10s}  {'--':>16s}")
            else:
                print(f"  {nx:6d}  {gv_sigma:10.4f}  {gv_err:16.6e}")
                if best_overall is None or gv_err < best_overall[2]:
                    best_overall = (nx, gv_sigma, gv_err)
        if best_overall is not None:
            nx_b, sigma_b, gv_b = best_overall
            print(
                f"\n  Best (nextra, sigma) by GV error: "
                f"nextra={nx_b}, sigma={sigma_b:.6f}, gv_err={gv_b:.6e}"
            )
        else:
            print("\n  Best (nextra, sigma) by GV error: (no feasible point)")

    return results, gv_by_nx_sigma


def run_nextra_penalty_sweep(
    n: int,
    nextra_values: list[int],
    n_sigma: int,
    n_gamma: int,
    sigma_max: float,
) -> dict[int, dict]:
    """Phase 2: nextra x sigma x gamma penalty sweep.

    For each nextra, sweeps (sigma, gamma) and reports the best point
    and gamma=0 baseline.
    Returns {nextra: {gamma0_se, gamma0_sigma, best_se, best_sigma, best_gamma, best_deficit}}.
    """
    sigmas = np.concatenate([
        [0.0],
        np.logspace(np.log10(0.01), np.log10(sigma_max), n_sigma),
    ])
    gammas = np.concatenate([
        [0.0],
        np.logspace(-1, 3, n_gamma),  # 0.1 to 1000
    ])

    results = {}
    for nx in nextra_values:
        r = Q + 1 + nx
        if n < 2 * r:
            continue

        best_gamma0_se = float("inf")
        best_gamma0_sigma = None
        best_se = float("inf")
        best_sigma = None
        best_gamma = None
        best_deficit = None

        for sigma in sigmas:
            for gamma in gammas:
                D = build_diff_matrix_rbf_penalty(
                    n, P, Q, sigma, "tension", NU, nx,
                    gamma=gamma,
                )
                se = stability_eigenvalue_from_matrix(D)
                deficit = float(np.max(np.abs(np.sum(D, axis=0))))

                if se < best_se:
                    best_se = se
                    best_sigma = sigma
                    best_gamma = gamma
                    best_deficit = deficit

                if gamma == 0.0 and se < best_gamma0_se:
                    best_gamma0_se = se
                    best_gamma0_sigma = sigma

        results[nx] = {
            "gamma0_se": best_gamma0_se,
            "gamma0_sigma": best_gamma0_sigma,
            "best_se": best_se,
            "best_sigma": best_sigma,
            "best_gamma": best_gamma,
            "best_deficit": best_deficit,
            "t": P + Q + 1 + nx,
            "r": Q + 1 + nx,
            "extra_dof": (Q + 1 + nx) * (P + nx),
        }

    # Print results
    total_per_nx = len(sigmas) * len(gammas)
    print(f"\n{'='*85}")
    print(f"  E4 Tension + Penalty — nextra x sigma x gamma Sweep (n={n})")
    print(f"{'='*85}")
    print(f"  Grid: {len(nextra_values)} nextra x {len(sigmas)} sigma x "
          f"{len(gammas)} gamma = {len(nextra_values) * total_per_nx} points")

    print(f"\n  {'nextra':>6s}  {'t':>3s}  {'r':>3s}  {'DOF':>5s}  "
          f"{'g=0 stab_eig':>14s}  {'best sigma':>10s}  {'best gamma':>10s}  "
          f"{'(s,g) stab_eig':>16s}  {'status':>10s}")
    print(f"  {'-'*6}  {'-'*3}  {'-'*3}  {'-'*5}  "
          f"{'-'*14}  {'-'*10}  {'-'*10}  "
          f"{'-'*16}  {'-'*10}")

    for nx in nextra_values:
        if nx not in results:
            continue
        res = results[nx]
        status = "STABLE" if res["best_se"] < STABILITY_TOL else "unstable"
        print(f"  {nx:6d}  {res['t']:3d}  {res['r']:3d}  {res['extra_dof']:5d}  "
              f"{res['gamma0_se']:14.6e}  {res['best_sigma']:10.4f}  "
              f"{res['best_gamma']:10.4f}  "
              f"{res['best_se']:16.6e}  {status:>10s}")

    for nx in nextra_values:
        if nx not in results:
            continue
        res = results[nx]
        print(f"\n  nextra={nx}: gamma=0 sigma*={res['gamma0_sigma']:.4f} "
              f"-> {res['gamma0_se']:.6e}")
        print(f"          : (sigma,gamma)*=({res['best_sigma']:.4f}, {res['best_gamma']:.4f}) "
              f"-> {res['best_se']:.6e}  deficit={res['best_deficit']:.6e}")

    return results


def run_grid_independence(
    nextra_values: list[int],
    grid_sizes: list[int],
) -> dict[int, list[int]]:
    """Phase 3: Grid independence check at sigma=0 (PHS k=2).

    Returns {nextra: [stable grid sizes]}.
    """
    print(f"\n{'='*60}")
    print(f"  E4 PHS k=2 (sigma=0) — Grid Independence")
    print(f"{'='*60}")
    print(f"  {'nextra':>6s}  {'n':>6s}  {'stab_eig':>14s}  {'status':>10s}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*14}  {'-'*10}")

    results = {}
    for nx in nextra_values:
        stable_at = []
        for nn in grid_sizes:
            r = Q + 1 + nx
            if nn < 2 * r:
                continue
            se = stability_eigenvalue(
                nn, p=P, q=Q, epsilon=0.0,
                kernel="tension", nu=NU, nextra=nx,
            )
            status = "STABLE" if se < STABILITY_TOL else "unstable"
            print(f"  {nx:6d}  {nn:6d}  {se:14.6e}  {status:>10s}")
            if se < STABILITY_TOL:
                stable_at.append(nn)
        results[nx] = stable_at

    return results


def _check_gv_sigma_stable_grids(
    nextra: int, gv_sigma: float, grid_sizes: list[int],
) -> list[int]:
    """Cross-grid stability re-check for a GV-optimal sigma at a given nextra.

    Mirrors the cross-grid re-check used by tension_sweep (40.2c) and
    epsilon_sweep (40.3c). Returns the list of grid sizes in ``grid_sizes``
    at which ``stability_eigenvalue`` is below ``STABILITY_TOL`` for the
    E4 tension kernel at ``(nextra, sigma)``.
    """
    r = Q + 1 + nextra
    stable_at: list[int] = []
    for nn in grid_sizes:
        if nn < 2 * r:
            continue
        se = stability_eigenvalue(
            nn, p=P, q=Q, epsilon=gv_sigma,
            kernel="tension", nu=NU, nextra=nextra,
        )
        status = "STABLE" if se < STABILITY_TOL else "unstable"
        print(f"    n={nn:3d}  stab_eig={se:14.6e}  [{status}]")
        if se < STABILITY_TOL:
            stable_at.append(nn)
    return stable_at


def run_footprint_sweep(
    n_sigma: int,
    n_gamma: int,
    sigma_max: float = 50.0,
    nextra_values: list[int] | None = None,
    *,
    include_gv: bool = False,
) -> dict:
    """Run all three phases and return summary for known_values.json.

    Returns dict with per-nextra stability info.
    """
    if nextra_values is None:
        nextra_values = [0, 1, 2, 3]
    n = 40  # primary grid size, matching test classes
    grid_sizes = [20, 40, 80, 160]

    # Phase 1: nextra x sigma
    sigma_results, gv_by_nx_sigma = run_nextra_sigma_sweep(
        n, nextra_values, n_sigma, sigma_max, include_gv=include_gv,
    )

    # Phase 2: nextra x sigma x gamma penalty
    penalty_results = run_nextra_penalty_sweep(
        n, nextra_values, n_sigma, n_gamma, sigma_max,
    )

    # Phase 3: grid independence at sigma=0 (PHS k=2)
    grid_results = run_grid_independence(nextra_values, grid_sizes)

    # Phase 4 (GV-only): cross-grid stability re-check at the GV-optimal sigma
    # for each nextra. Mirrors 40.2c/40.3c/40.4d.
    gv_stable_at_by_nx: dict[int, list[int]] = {}
    if include_gv:
        # Filter sigma > 0 (40.5d): a "tension" GV optimum at sigma=0 is
        # really the PHS k=2 baseline, not a tension-kernel point. The
        # eigenvalue-stability path applies the same filter at line 440.
        any_gv = any(
            (sigma_results.get(nx, {}).get("best_stable_gv_sigma") or 0.0) > 0
            for nx in nextra_values
        )
        if any_gv:
            print(f"\n{'='*70}")
            print("  Cross-grid stability re-check at GV-optimal sigma (per nextra)")
            print(f"{'='*70}")
        for nx in nextra_values:
            gv_sigma = sigma_results.get(nx, {}).get("best_stable_gv_sigma")
            if gv_sigma is None or gv_sigma <= 0:
                continue
            print(f"  nextra={nx}, sigma={gv_sigma:.6f}:")
            gv_stable_at_by_nx[nx] = _check_gv_sigma_stable_grids(
                nx, gv_sigma, grid_sizes,
            )

    # Build summary
    summary = {}
    for nx in nextra_values:
        key = f"E4_nextra{nx}_phs"
        entry = {"nextra": nx}
        if nx in grid_results:
            entry["stable_at"] = grid_results[nx]
        summary[key] = entry

        # Also record best tension sigma if nextra is in sigma_results
        if nx in sigma_results:
            res = sigma_results[nx]
            if res["best_se"] < STABILITY_TOL and res["best_sigma"] > 0:
                best_sigma_rounded = round(float(res["best_sigma"]), 4)
                t_key = f"E4_nextra{nx}_tension_{res['best_sigma']:.0f}"
                t_entry: dict = {
                    "nextra": nx,
                    "sigma": best_sigma_rounded,
                    "stable_at": [n],
                }
                # The additive ``gv_error`` field must represent the GV at
                # the stability-optimum sigma (``best_sigma``), not at the
                # GV-optimum sigma (``best_stable_gv_sigma``) — otherwise
                # the ``(sigma, gv_error)`` pair describes two different
                # points. Evaluate at the rounded sigma so the persisted
                # pair is exactly self-consistent. The GV at the GV-optimum
                # sigma is persisted on the parallel
                # ``E4_nextra{nx}_tension_gv`` entry below.
                if include_gv:
                    t_entry["gv_error"] = float(boundary_gv_error_max(
                        p=P, q=Q, nextra=nx, nu=NU,
                        sigma=best_sigma_rounded, kernel="tension",
                    ))
                summary[t_key] = t_entry

            # Parallel GV-optimal entry (separate from the stability-optimum
            # key because the sigma is embedded in the key name). Filter
            # sigma > 0 (40.5d) so a GV optimum that collapses to the PHS
            # k=2 baseline is not mis-labeled as a tension entry — that
            # information already lives on ``E4_nextra{nx}_phs``.
            if (
                include_gv
                and res.get("best_stable_gv_sigma") is not None
                and res["best_stable_gv_sigma"] > 0
            ):
                gv_sigma_rounded = round(float(res["best_stable_gv_sigma"]), 4)
                gv_key = f"E4_nextra{nx}_tension_gv"
                summary[gv_key] = {
                    "nextra": nx,
                    "sigma": gv_sigma_rounded,
                    "gv_error": float(boundary_gv_error_max(
                        p=P, q=Q, nextra=nx, nu=NU,
                        sigma=gv_sigma_rounded, kernel="tension",
                    )),
                    "stable_at": gv_stable_at_by_nx.get(nx, []),
                }

    if gv_by_nx_sigma is not None:
        summary["_gv_by_nx_sigma"] = gv_by_nx_sigma

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sweeps.footprint_sweep",
        description="Stencil footprint (nextra) sweep for E4 tension kernel",
    )
    parser.add_argument(
        "--n-sigma", type=int, default=20,
        help="Number of sigma sample points (default: 20)",
    )
    parser.add_argument(
        "--n-gamma", type=int, default=20,
        help="Number of gamma sample points for penalty phase (default: 20)",
    )
    parser.add_argument(
        "--sigma-max", type=float, default=50.0,
        help="Maximum sigma value (default: 50.0)",
    )
    parser.add_argument(
        "--nextra-values", default="0,1,2,3",
        help="Comma-separated nextra values (default: 0,1,2,3)",
    )
    parser.add_argument(
        "--update-known-values", action="store_true",
        help="Update known_values.json with discovered footprint stability",
    )
    parser.add_argument(
        "--include-gv", action="store_true",
        help="Also compute boundary group-velocity error per (nextra, sigma) (advisory)",
    )

    args = parser.parse_args(argv)
    nextra_values = [int(x) for x in args.nextra_values.split(",")]

    summary = run_footprint_sweep(
        args.n_sigma, args.n_gamma, args.sigma_max, nextra_values,
        include_gv=args.include_gv,
    )

    if args.update_known_values:
        kv = load_known_values()
        footprint = kv.setdefault("footprint", {})
        # Per-entry additive merge (pattern from 40.2d/40.3c/40.4c): read
        # the existing entry dict, update with only the keys this invocation
        # owns, and reassign. This preserves pre-existing keys (e.g. a
        # ``gv_error`` set by a prior ``--include-gv`` run) when the current
        # invocation does not set them, and avoids the pre-existing bug where
        # ``kv["footprint"] = summary`` silently dropped entries keyed by a
        # sigma value this run did not reproduce. Drop internal keys (leading
        # underscore) that are for in-process use only.
        for key, new_entry in summary.items():
            if key.startswith("_"):
                continue
            merged = dict(footprint.get(key, {}))
            merged.update(new_entry)
            footprint[key] = merged
        save_known_values(kv)
        print(f"\n  Updated known_values.json: footprint")

    return 0


if __name__ == "__main__":
    sys.exit(main())
