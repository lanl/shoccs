"""CLI entry point for sweep scripts.

Usage:
    uv run python -m sweeps epsilon --scheme E2
    uv run python -m sweeps tension --scheme E4
    uv run python -m sweeps all --quick
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="sweeps",
        description="Parameter-space exploration sweeps for PHS/RBF stencils",
    )
    subparsers = parser.add_subparsers(dest="command", help="sweep type")

    # Placeholder subcommands — implementations added as sweep scripts land
    sub_eps = subparsers.add_parser("epsilon", help="Epsilon (Gaussian/MQ) sweep")
    sub_eps.add_argument("--scheme", choices=["E2", "E4"], required=True)
    sub_eps.add_argument("--kernel", choices=["gaussian", "multiquadric"], default="gaussian")
    sub_eps.add_argument("--n-values", default="20,40,80", help="Comma-separated grid sizes")
    sub_eps.add_argument("--n-eps", type=int, default=60, help="Number of epsilon sample points")
    sub_eps.add_argument("--update-known-values", action="store_true", help="Update known_values.json with discovered optimal epsilon")
    sub_eps.add_argument("--include-gv", action="store_true", help="Also compute boundary group-velocity error at each epsilon (advisory)")
    sub_eps.add_argument("--check-gks", action="store_true", help="Run gks_group_velocity_check on D at eps* and print outgoing-mode WARNINGs (advisory)")
    from .epsilon_sweep import CLI_DEFAULT_EPS_FLOOR

    sub_eps.add_argument("--eps-floor", type=float, default=CLI_DEFAULT_EPS_FLOOR, help=f"Restrict fine-sweep search to epsilon >= eps_floor (default {CLI_DEFAULT_EPS_FLOOR}) so the persisted gaussian entry stays strictly above the eps -> 0 degenerate-kernel limit; see plan 46.3b.1.2")

    sub_tension = subparsers.add_parser("tension", help="Tension spline sigma sweep")
    sub_tension.add_argument("--scheme", choices=["E2", "E4"], required=True)
    sub_tension.add_argument("--n-values", default="20,40,80", help="Comma-separated grid sizes")
    sub_tension.add_argument("--n-sigma", type=int, default=61, help="Number of sigma sample points")
    sub_tension.add_argument("--sigma-max", type=float, default=20.0)
    sub_tension.add_argument("--update-known-values", action="store_true", help="Update known_values.json with discovered optimal sigma")
    sub_tension.add_argument("--include-gv", action="store_true", help="Also compute boundary group-velocity error at each sigma (advisory)")
    sub_tension.add_argument("--check-gks", action="store_true", help="Run gks_group_velocity_check on D at sigma* and print outgoing-mode WARNINGs (advisory)")
    from .tension_sweep import CLI_DEFAULT_SIGMA_FLOOR

    sub_tension.add_argument("--sigma-floor", type=float, default=CLI_DEFAULT_SIGMA_FLOOR, help=f"Restrict fine-sweep search to sigma >= sigma_floor (default {CLI_DEFAULT_SIGMA_FLOOR}) so the persisted tension entry stays strictly above the PHS k=2 limit (sigma=0); see plan 46.3a.2")

    sub_tension_pen = subparsers.add_parser("tension-penalty", help="Tension + conservation penalty sweep")
    sub_tension_pen.add_argument("--scheme", choices=["E2", "E4"], required=True)
    sub_tension_pen.add_argument("--n-sigma", type=int, default=25)
    sub_tension_pen.add_argument("--n-gamma", type=int, default=25)
    sub_tension_pen.add_argument("--sigma-max", type=float, default=20.0)
    sub_tension_pen.add_argument("--update-known-values", action="store_true", help="Update known_values.json")

    sub_footprint = subparsers.add_parser("footprint", help="Stencil footprint (nextra) sweep")
    sub_footprint.add_argument("--n-sigma", type=int, default=20)
    sub_footprint.add_argument("--n-gamma", type=int, default=20)
    sub_footprint.add_argument("--sigma-max", type=float, default=50.0)
    sub_footprint.add_argument("--nextra-values", default="0,1,2,3", help="Comma-separated nextra values")
    sub_footprint.add_argument("--update-known-values", action="store_true", help="Update known_values.json")
    sub_footprint.add_argument("--include-gv", action="store_true", help="Also compute boundary group-velocity error per (nextra, sigma) (advisory)")

    sub_comparison = subparsers.add_parser("comparison", help="Multi-method comparison table")
    sub_comparison.add_argument("--scheme", choices=["E2", "E4"], default=None)
    sub_comparison.add_argument("--n-values", default="20,40,80", help="Comma-separated grid sizes")
    sub_comparison.add_argument("--update-known-values", action="store_true", help="Update known_values.json")

    sub_alpha = subparsers.add_parser("alpha", help="Boundary alpha extraction at optimal epsilon")
    sub_alpha.add_argument("--scheme", choices=["E2", "E4"], required=True)

    sub_mixed = subparsers.add_parser("mixed-epsilon", help="Mixed (per-row) epsilon sweep")
    sub_mixed.add_argument("--scheme", choices=["E2", "E4"], default="E4")
    sub_mixed.add_argument("--kernel", choices=["gaussian", "multiquadric"], default="gaussian")
    sub_mixed.add_argument("--n-eps", type=int, default=20)
    sub_mixed.add_argument("--update-known-values", action="store_true", help="Update known_values.json")

    sub_pareto = subparsers.add_parser(
        "gv-stability-pareto",
        help=(
            "GV-vs-stability 1D parametric Pareto scan (research / documentation aid). "
            "For NSGA-II multi-objective optimization across 2+ stability metrics, "
            "use the 'pareto' subcommand instead."
        ),
    )
    sub_pareto.add_argument("--scheme", choices=["E2", "E4"], required=True)
    sub_pareto.add_argument(
        "--param", choices=["tension", "gaussian", "multiquadric"], required=True,
        help="Kernel to sweep (parameter is sigma for tension, epsilon otherwise)",
    )
    sub_pareto.add_argument("--n-points", type=int, default=61, help="Number of sample points on the parameter grid")
    sub_pareto.add_argument("--n", type=int, default=40, help="Grid size for the stability eigenvalue")
    sub_pareto.add_argument("--param-max", type=float, default=20.0, help="Upper end of the parameter grid")

    sub_pareto_nsga = subparsers.add_parser(
        "pareto",
        help=(
            "NSGA-II multi-objective Pareto front over Brady-Livescu 2D stability "
            "metrics. Distinct from 'gv-stability-pareto' (a 1D parametric scan)."
        ),
    )
    sub_pareto_nsga.add_argument("--scheme", choices=["E2", "E4"], required=True)
    sub_pareto_nsga.add_argument(
        "--kernel",
        choices=["tension", "gaussian", "multiquadric", "classical"],
        required=True,
    )
    sub_pareto_nsga.add_argument(
        "--objectives",
        nargs="+",
        required=True,
        metavar="FIELD",
        help='Two or more dotted-path report fields (e.g. "layer1.boundary_gv_err layer_bl42.max_spectral_abscissa").',
    )
    sub_pareto_nsga.add_argument(
        "--bounds",
        type=float,
        nargs="+",
        default=None,
        metavar="VAL",
        help="Flat list of bound pairs (lo hi [lo hi ...]). Falls back to DEFAULT_BOUNDS if absent.",
    )
    sub_pareto_nsga.add_argument("--pop-size", type=int, default=40)
    sub_pareto_nsga.add_argument("--n-gen", type=int, default=50)
    sub_pareto_nsga.add_argument("--seed", type=int, default=1)
    sub_pareto_nsga.add_argument(
        "--ref-point",
        type=float,
        nargs="+",
        default=None,
        metavar="V",
        help="Reference point for the hypervolume indicator (one value per --objectives).",
    )
    sub_pareto_nsga.add_argument(
        "--gate-layer",
        type=int,
        default=None,
        help="Highest layer whose failure forces the sentinel vector. Default: max_layer-1.",
    )
    sub_pareto_nsga.add_argument(
        "--max-layer",
        type=int,
        default=None,
        help="Highest layer executed per evaluation. Default: max(_infer_max_layer(f) for f in --objectives).",
    )
    sub_pareto_nsga.add_argument(
        "--persist",
        action="store_true",
        help="Persist the ParetoResult as JSON under sweeps/pareto_fronts/.",
    )
    sub_pareto_nsga.add_argument(
        "--validate-with-cpp",
        action="store_true",
        help="Re-run up to 10 front members at max_layer=8 via the C++ bridge.",
    )
    sub_pareto_nsga.add_argument(
        "--verbose",
        action="store_true",
        help="Forward to pymoo's minimize(verbose=True).",
    )

    sub_brady2d = subparsers.add_parser(
        "brady2d",
        help="Brady-Livescu 2D layered stability sweep",
    )
    sub_brady2d.add_argument("--scheme", choices=["E2", "E4"], required=True)
    sub_brady2d.add_argument(
        "--kernel",
        choices=["classical", "tension", "gaussian", "multiquadric"],
        required=True,
    )
    sub_brady2d.add_argument(
        "--param-range",
        nargs=3,
        metavar=("LO", "HI", "N"),
        default=None,
        help=(
            "Three values: low, high, and number of sweep points. "
            "Sweeps sigma for tension, epsilon for gaussian/multiquadric. "
            "Ignored when --kernel=classical."
        ),
    )
    sub_brady2d.add_argument(
        "--max-layer", type=int, default=3, choices=list(range(1, 9)),
        help="Highest stability layer to run (1-8). Default: 3.",
    )
    sub_brady2d.add_argument(
        "--validate-with-cpp", action="store_true",
        help="Re-run top-3 passing points at max_layer=8 via the C++ bridge.",
    )
    sub_brady2d.add_argument(
        "--layer8-N", type=int, default=21,
        help="Grid resolution forwarded to layer 8 (default: 21).",
    )
    sub_brady2d.add_argument(
        "--layer8-t-final", type=float, default=1.0,
        help="Simulation end time forwarded to layer 8 (default: 1.0).",
    )
    sub_brady2d.add_argument(
        "--update-known-values", action="store_true",
        help='Persist results to known_values.json["brady2d_sweep"][scheme][kernel].',
    )

    sub_opt = subparsers.add_parser(
        "optimize",
        help="Optimize boundary-closure parameters against a stability objective",
    )
    sub_opt.add_argument("--scheme", choices=["E2", "E4"], required=True)
    sub_opt.add_argument(
        "--kernel",
        choices=["tension", "gaussian", "multiquadric", "classical"],
        required=True,
    )
    sub_opt.add_argument(
        "--objective",
        required=True,
        help='Dotted-path report field (e.g. "layer3.max_stab_eig", "layer6.transient_growth_bound").',
    )
    sub_opt.add_argument(
        "--gate-layer",
        type=int,
        default=None,
        help="Layer N where failure short-circuits to +inf. Default: max(--max-layer - 1, 0) auto-inferred from objective.",
    )
    sub_opt.add_argument("--max-layer", type=int, default=None)
    sub_opt.add_argument(
        "--bounds",
        type=float,
        nargs="+",
        default=None,
        metavar="VAL",
        help="Flat list of bound pairs (lo hi [lo hi ...]). Falls back to DEFAULT_BOUNDS if absent.",
    )
    sub_opt.add_argument(
        "--method",
        choices=["Nelder-Mead", "COBYQA", "SHGO", "DE", "staged"],
        default="staged",
    )
    sub_opt.add_argument("--n-restarts", type=int, default=10)
    sub_opt.add_argument("--max-evals", type=int, default=200)
    sub_opt.add_argument("--seed", type=int, default=0)
    sub_opt.add_argument("--validator-max-layer", type=int, default=6)
    sub_opt.add_argument("--top-k", type=int, default=5)
    sub_opt.add_argument(
        "--inner-method",
        choices=["Nelder-Mead", "COBYQA"],
        default="Nelder-Mead",
    )
    sub_opt.add_argument("--shgo-n", type=int, default=100)
    sub_opt.add_argument("--shgo-iters", type=int, default=3)
    sub_opt.add_argument("--de-popsize", type=int, default=15)
    sub_opt.add_argument("--de-maxiter", type=int, default=100)
    sub_opt.add_argument("--validate-with-cpp", action="store_true")
    sub_opt.add_argument("--update-known-values", action="store_true")
    sub_opt.add_argument("--json-output", type=str, default=None)

    sub_bo = subparsers.add_parser(
        "bo",
        help="Multi-fidelity Bayesian optimization (BoTorch qMFKG)",
    )
    sub_bo.add_argument("--scheme", choices=["E2", "E4"], required=True)
    sub_bo.add_argument(
        "--kernel",
        choices=["tension", "gaussian", "multiquadric", "classical"],
        required=True,
    )
    sub_bo.add_argument(
        "--objective",
        required=True,
        help='HF target as a dotted-path report field, e.g. "layer7.max_spectral_abscissa".',
    )
    sub_bo.add_argument(
        "--cheap-fidelities",
        type=int,
        nargs="+",
        required=True,
        metavar="N",
        help="External cascade layer indices to use as cheap surrogates (each < HF layer).",
    )
    sub_bo.add_argument(
        "--fidelity-fields",
        nargs="+",
        default=None,
        metavar="LAYER=FIELD",
        help="Per-layer field overrides (e.g. '3=layer3.something_else').",
    )
    sub_bo.add_argument(
        "--bounds",
        type=float,
        nargs="+",
        default=None,
        metavar="VAL",
        help="Flat list of bound pairs (lo hi [lo hi ...]). Falls back to DEFAULT_BOUNDS if absent.",
    )
    bo_budget = sub_bo.add_mutually_exclusive_group(required=True)
    bo_budget.add_argument(
        "--budget-evals",
        type=int,
        default=None,
        help="Total number of cascade evaluations (init + acquisition + final HF).",
    )
    bo_budget.add_argument(
        "--budget-seconds",
        type=float,
        default=None,
        help="Wall-time budget in seconds (mutually exclusive with --budget-evals).",
    )
    sub_bo.add_argument(
        "--n-init",
        type=int,
        default=None,
        help="Initial design size (default: 5*d + 3 per Loeppky et al. 2009).",
    )
    sub_bo.add_argument(
        "--num-fantasies",
        type=int,
        default=64,
        help="Number of fantasies for qMFKG (default: 64).",
    )
    sub_bo.add_argument("--seed", type=int, default=1)
    sub_bo.add_argument(
        "--cost-model",
        choices=["constant", "empirical"],
        default="constant",
        help="'constant' uses the plan-46 calibrated DEFAULT_COST_TABLE.",
    )
    sub_bo.add_argument(
        "--baseline",
        choices=["none", "staged"],
        default="none",
        help="Run a comparator alongside MF-BO with the same seed (plan 47.5b).",
    )
    sub_bo.add_argument(
        "--persist",
        action="store_true",
        help="Persist the BOResult as JSON under sweeps/bo_runs/ (plan 47.4c).",
    )
    sub_bo.add_argument(
        "--validate-with-cpp",
        action="store_true",
        help="Re-run best_x at max_layer=8 via the C++ bridge (plan 47.5a).",
    )
    sub_bo.add_argument(
        "--verbose",
        action="store_true",
        help="Forward to run_mfbo(verbose=True): one line per evaluation.",
    )

    sub_all = subparsers.add_parser("all", help="Run all sweeps")
    sub_all.add_argument("--quick", action="store_true", help="Reduced resolution for quick verification")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Dispatch to sweep modules (imported lazily to avoid loading numpy/sympy at parse time)
    if args.command == "epsilon":
        from .epsilon_sweep import main as eps_main

        return eps_main([
            "--scheme", args.scheme,
            "--kernel", args.kernel,
            "--n-values", args.n_values,
            "--n-eps", str(args.n_eps),
            "--eps-floor", str(args.eps_floor),
            *(["--update-known-values"] if args.update_known_values else []),
            *(["--include-gv"] if args.include_gv else []),
            *(["--check-gks"] if args.check_gks else []),
        ])

    if args.command == "tension":
        from .tension_sweep import main as tension_main

        return tension_main([
            "--scheme", args.scheme,
            "--n-values", args.n_values,
            "--n-sigma", str(args.n_sigma),
            "--sigma-max", str(args.sigma_max),
            "--sigma-floor", str(args.sigma_floor),
            *(["--update-known-values"] if args.update_known_values else []),
            *(["--include-gv"] if args.include_gv else []),
            *(["--check-gks"] if args.check_gks else []),
        ])

    if args.command == "mixed-epsilon":
        from .mixed_epsilon_sweep import main as mixed_main

        return mixed_main([
            "--scheme", args.scheme,
            "--kernel", args.kernel,
            "--n-eps", str(args.n_eps),
            *(["--update-known-values"] if args.update_known_values else []),
        ])

    if args.command == "tension-penalty":
        from .tension_penalty_sweep import main as tp_main

        return tp_main([
            "--scheme", args.scheme,
            "--n-sigma", str(args.n_sigma),
            "--n-gamma", str(args.n_gamma),
            "--sigma-max", str(args.sigma_max),
            *(["--update-known-values"] if args.update_known_values else []),
        ])

    if args.command == "footprint":
        from .footprint_sweep import main as fp_main

        return fp_main([
            "--n-sigma", str(args.n_sigma),
            "--n-gamma", str(args.n_gamma),
            "--sigma-max", str(args.sigma_max),
            "--nextra-values", args.nextra_values,
            *(["--update-known-values"] if args.update_known_values else []),
            *(["--include-gv"] if args.include_gv else []),
        ])

    if args.command == "comparison":
        from .comparison import main as comp_main

        return comp_main([
            *(["--scheme", args.scheme] if args.scheme else []),
            "--n-values", args.n_values,
            *(["--update-known-values"] if args.update_known_values else []),
        ])

    if args.command == "alpha":
        from .alpha_extraction import main as alpha_main

        return alpha_main(["--scheme", args.scheme])

    if args.command == "gv-stability-pareto":
        from .gv_stability_pareto import main as pareto_main

        return pareto_main([
            "--scheme", args.scheme,
            "--param", args.param,
            "--n-points", str(args.n_points),
            "--n", str(args.n),
            "--param-max", str(args.param_max),
        ])

    if args.command == "pareto":
        from .pareto import main as pareto_nsga_main

        forwarded: list[str] = [
            "--scheme", args.scheme,
            "--kernel", args.kernel,
            "--objectives", *args.objectives,
            "--pop-size", str(args.pop_size),
            "--n-gen", str(args.n_gen),
            "--seed", str(args.seed),
        ]
        if args.bounds is not None:
            forwarded.append("--bounds")
            forwarded.extend(str(v) for v in args.bounds)
        if args.ref_point is not None:
            forwarded.append("--ref-point")
            forwarded.extend(str(v) for v in args.ref_point)
        if args.gate_layer is not None:
            forwarded.extend(["--gate-layer", str(args.gate_layer)])
        if args.max_layer is not None:
            forwarded.extend(["--max-layer", str(args.max_layer)])
        if args.persist:
            forwarded.append("--persist")
        if args.validate_with_cpp:
            forwarded.append("--validate-with-cpp")
        if args.verbose:
            forwarded.append("--verbose")
        return pareto_nsga_main(forwarded)

    if args.command == "brady2d":
        from .brady2d_sweep import main as brady2d_main

        forwarded: list[str] = [
            "--scheme", args.scheme,
            "--kernel", args.kernel,
            "--max-layer", str(args.max_layer),
            "--layer8-N", str(args.layer8_N),
            "--layer8-t-final", str(args.layer8_t_final),
        ]
        if args.param_range is not None:
            forwarded.extend(["--param-range", *[str(v) for v in args.param_range]])
        if args.validate_with_cpp:
            forwarded.append("--validate-with-cpp")
        if args.update_known_values:
            forwarded.append("--update-known-values")
        return brady2d_main(forwarded)

    if args.command == "optimize":
        from .optimize import main as optimize_main

        forwarded: list[str] = [
            "--scheme", args.scheme,
            "--kernel", args.kernel,
            "--objective", args.objective,
            "--method", args.method,
            "--n-restarts", str(args.n_restarts),
            "--max-evals", str(args.max_evals),
            "--seed", str(args.seed),
            "--validator-max-layer", str(args.validator_max_layer),
            "--top-k", str(args.top_k),
            "--inner-method", args.inner_method,
            "--shgo-n", str(args.shgo_n),
            "--shgo-iters", str(args.shgo_iters),
            "--de-popsize", str(args.de_popsize),
            "--de-maxiter", str(args.de_maxiter),
        ]
        if args.gate_layer is not None:
            forwarded.extend(["--gate-layer", str(args.gate_layer)])
        if args.max_layer is not None:
            forwarded.extend(["--max-layer", str(args.max_layer)])
        if args.bounds is not None:
            forwarded.append("--bounds")
            forwarded.extend(str(v) for v in args.bounds)
        if args.validate_with_cpp:
            forwarded.append("--validate-with-cpp")
        if args.update_known_values:
            forwarded.append("--update-known-values")
        if args.json_output is not None:
            forwarded.extend(["--json-output", args.json_output])
        return optimize_main(forwarded)

    if args.command == "bo":
        from .bo import main as bo_main

        forwarded: list[str] = [
            "--scheme", args.scheme,
            "--kernel", args.kernel,
            "--objective", args.objective,
            "--cheap-fidelities", *[str(v) for v in args.cheap_fidelities],
            "--num-fantasies", str(args.num_fantasies),
            "--seed", str(args.seed),
            "--cost-model", args.cost_model,
            "--baseline", args.baseline,
        ]
        if args.fidelity_fields is not None:
            forwarded.extend(["--fidelity-fields", *args.fidelity_fields])
        if args.bounds is not None:
            forwarded.append("--bounds")
            forwarded.extend(str(v) for v in args.bounds)
        if args.budget_evals is not None:
            forwarded.extend(["--budget-evals", str(args.budget_evals)])
        if args.budget_seconds is not None:
            forwarded.extend(["--budget-seconds", str(args.budget_seconds)])
        if args.n_init is not None:
            forwarded.extend(["--n-init", str(args.n_init)])
        if args.persist:
            forwarded.append("--persist")
        if args.validate_with_cpp:
            forwarded.append("--validate-with-cpp")
        if args.verbose:
            forwarded.append("--verbose")
        return bo_main(forwarded)

    if args.command == "all":
        return _run_all(quick=args.quick)

    print(f"sweeps: command '{args.command}' not recognized")
    return 1


def _run_all(*, quick: bool) -> int:
    """Run all sweeps sequentially. --quick reduces resolution for fast verification."""
    from .alpha_extraction import main as alpha_main
    from .brady2d_sweep import main as brady2d_main
    from .comparison import main as comp_main
    from .epsilon_sweep import main as eps_main
    from .footprint_sweep import main as fp_main
    from .gv_stability_pareto import main as pareto_main
    from .mixed_epsilon_sweep import main as mixed_main
    from .tension_penalty_sweep import main as tp_main
    from .tension_sweep import main as tension_main

    quick_n_eps = "10" if quick else "60"
    quick_n_sigma = "10" if quick else "61"
    quick_n_gamma = "5" if quick else "25"
    quick_n_values = "20,40" if quick else "20,40,80"
    quick_mixed_n_eps = "5" if quick else "20"
    quick_fp_n_sigma = "10" if quick else "20"
    quick_fp_n_gamma = "10" if quick else "20"
    quick_tp_n_sigma = "5" if quick else "25"
    quick_pareto_n_points = "10" if quick else "61"

    gv_flag = ["--include-gv"] if quick else []

    sweeps: list[tuple[str, Callable[[list[str]], int], list[str]]] = [
        ("Epsilon sweep E2 (gaussian)", eps_main,
         ["--scheme", "E2", "--kernel", "gaussian", "--n-eps", quick_n_eps, "--n-values", quick_n_values, *gv_flag]),
        ("Epsilon sweep E4 (gaussian)", eps_main,
         ["--scheme", "E4", "--kernel", "gaussian", "--n-eps", quick_n_eps, "--n-values", quick_n_values]),
        ("Epsilon sweep E2 (multiquadric)", eps_main,
         ["--scheme", "E2", "--kernel", "multiquadric", "--n-eps", quick_n_eps, "--n-values", quick_n_values]),
        ("Epsilon sweep E4 (multiquadric)", eps_main,
         ["--scheme", "E4", "--kernel", "multiquadric", "--n-eps", quick_n_eps, "--n-values", quick_n_values]),
        ("Mixed epsilon sweep E4", mixed_main,
         ["--scheme", "E4", "--n-eps", quick_mixed_n_eps]),
        ("Tension sweep E2", tension_main,
         ["--scheme", "E2", "--n-sigma", quick_n_sigma, "--n-values", quick_n_values, *gv_flag]),
        ("Tension sweep E4", tension_main,
         ["--scheme", "E4", "--n-sigma", quick_n_sigma, "--n-values", quick_n_values, *gv_flag]),
        ("Tension-penalty sweep E2", tp_main,
         ["--scheme", "E2", "--n-sigma", quick_tp_n_sigma, "--n-gamma", quick_n_gamma]),
        ("Tension-penalty sweep E4", tp_main,
         ["--scheme", "E4", "--n-sigma", quick_tp_n_sigma, "--n-gamma", quick_n_gamma]),
        ("Footprint sweep", fp_main,
         ["--n-sigma", quick_fp_n_sigma, "--n-gamma", quick_fp_n_gamma, *gv_flag]),
        ("Comparison (all schemes)", comp_main,
         ["--n-values", quick_n_values]),
        ("GV-stability Pareto E2 (tension)", pareto_main,
         ["--scheme", "E2", "--param", "tension", "--n-points", quick_pareto_n_points]),
        ("Alpha extraction E2", alpha_main, ["--scheme", "E2"]),
        ("Brady-Livescu 2D sweep E4 (tension)", brady2d_main,
         ["--scheme", "E4", "--kernel", "tension", "--param-range", "2", "4", "3", "--max-layer", "3"]),
    ]

    failures: list[str] = []
    for label, fn, argv in sweeps:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}\n")
        try:
            rc = fn(argv)
            if rc != 0:
                print(f"\n*** {label} returned exit code {rc}")
                failures.append(label)
        except Exception as exc:
            print(f"\n*** {label} failed: {exc}")
            failures.append(label)

    print(f"\n{'=' * 60}")
    if failures:
        print(f"  {len(failures)} sweep(s) failed:")
        for f in failures:
            print(f"    - {f}")
        print(f"{'=' * 60}")
        return 1
    print(f"  All {len(sweeps)} sweeps completed successfully")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
