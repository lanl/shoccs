#!/usr/bin/env python3
"""Compare two Google Benchmark JSON result files and report regressions.

Usage:
    bench_compare.py baseline.json current.json [--threshold PCT]

Matches benchmarks by name, computes the percentage change in real_time,
and prints a table.  Exits non-zero if any benchmark regresses beyond the
threshold (default 10%).
"""

import argparse
import json
import sys


def load_benchmarks(path):
    """Return a dict mapping benchmark name -> record from a Google Benchmark JSON file."""
    with open(path) as f:
        data = json.load(f)
    return {b["name"]: b for b in data.get("benchmarks", [])}


def compare(baseline_path, current_path, threshold_pct):
    baseline = load_benchmarks(baseline_path)
    current = load_benchmarks(current_path)

    common = sorted(set(baseline) & set(current))
    only_baseline = sorted(set(baseline) - set(current))
    only_current = sorted(set(current) - set(baseline))

    if not common:
        print("No matching benchmarks found between the two files.")
        return 1

    regressions = []

    # Header
    print(f"{'Benchmark':<55} {'Baseline':>12} {'Current':>12} {'Change':>8}")
    print("-" * 91)

    for name in common:
        b_time = baseline[name]["real_time"]
        c_time = current[name]["real_time"]
        unit = baseline[name].get("time_unit", "ns")

        if b_time > 0:
            pct = (c_time - b_time) / b_time * 100.0
        else:
            pct = 0.0

        marker = ""
        if pct > threshold_pct:
            marker = " REGRESSION"
            regressions.append((name, pct))
        elif pct < -threshold_pct:
            marker = " improved"

        print(
            f"{name:<55} {b_time:>10.1f}{unit:>2} {c_time:>10.1f}{unit:>2} {pct:>+7.1f}%{marker}"
        )

    if only_baseline:
        print(f"\nRemoved ({len(only_baseline)}):")
        for name in only_baseline:
            print(f"  - {name}")

    if only_current:
        print(f"\nNew ({len(only_current)}):")
        for name in only_current:
            print(f"  + {name}")

    print()
    if regressions:
        print(f"FAIL: {len(regressions)} benchmark(s) regressed >{threshold_pct}%:")
        for name, pct in regressions:
            print(f"  {name}: {pct:+.1f}%")
        return 1

    print(f"PASS: no regressions above {threshold_pct}% threshold.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="Baseline JSON file")
    parser.add_argument("current", help="Current JSON file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Regression threshold in percent (default: 10)",
    )
    args = parser.parse_args()

    sys.exit(compare(args.baseline, args.current, args.threshold))


if __name__ == "__main__":
    main()
