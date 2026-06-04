#!/bin/bash
# Run the benchmark suite and compare results against a baseline.
#
# Usage:
#   ./scripts/bench_compare.sh                      # run all, compare vs stored baselines
#   ./scripts/bench_compare.sh baseline.json        # compare against a specific baseline
#   ./scripts/bench_compare.sh --save               # run and save as new baseline
#
# Prerequisites:
#   Build with -DBUILD_BENCHMARKS=ON
#
# The script runs every bench_* executable in build/benchmarks/, merges the
# results into a single JSON file, and compares against the baseline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${BUILD_DIR:-build}"
BENCH_DIR="${BUILD_DIR}/benchmarks"
BASELINE_DIR="${SCRIPT_DIR}/../benchmarks/baselines"
THRESHOLD="${BENCH_THRESHOLD:-10}"

# --- Parse arguments ---
SAVE_MODE=false
BASELINE_FILE=""

for arg in "$@"; do
    case "$arg" in
        --save)
            SAVE_MODE=true
            ;;
        *.json)
            BASELINE_FILE="$arg"
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            echo "Usage: $0 [--save] [baseline.json]" >&2
            exit 1
            ;;
    esac
done

# --- Discover benchmark executables ---
BENCHMARKS=()
for exe in "${BENCH_DIR}"/bench_*; do
    [[ -x "$exe" ]] && BENCHMARKS+=("$exe")
done

if [[ ${#BENCHMARKS[@]} -eq 0 ]]; then
    echo "Error: no benchmark executables found in ${BENCH_DIR}/" >&2
    echo "Build with: cmake -DBUILD_BENCHMARKS=ON .. && cmake --build ." >&2
    exit 1
fi

echo "Found ${#BENCHMARKS[@]} benchmark(s):"
printf "  %s\n" "${BENCHMARKS[@]}"
echo

# --- Run benchmarks and collect JSON ---
RESULTS_DIR=$(mktemp -d)
trap 'rm -rf "$RESULTS_DIR"' EXIT

for exe in "${BENCHMARKS[@]}"; do
    name=$(basename "$exe")
    out="${RESULTS_DIR}/${name}.json"
    echo "Running ${name}..."
    "$exe" --benchmark_out="$out" --benchmark_out_format=json 2>&1 | tail -1
done

# --- Merge into a single JSON file ---
MERGED="${RESULTS_DIR}/merged.json"
python3 -c "
import json, glob, sys

merged = {'benchmarks': []}
for path in sorted(glob.glob('${RESULTS_DIR}/bench_*.json')):
    with open(path) as f:
        data = json.load(f)
    # Keep context from the first file
    if 'context' not in merged:
        merged['context'] = data.get('context', {})
    merged['benchmarks'].extend(data.get('benchmarks', []))

with open('${MERGED}', 'w') as f:
    json.dump(merged, f, indent=2)
print(f'Merged {len(merged[\"benchmarks\"])} benchmark results.')
"

# --- Save or compare ---
if $SAVE_MODE; then
    mkdir -p "$BASELINE_DIR"
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    DEST="${BASELINE_DIR}/baseline-${TIMESTAMP}.json"
    cp "$MERGED" "$DEST"
    # Also update the "latest" symlink
    ln -sf "baseline-${TIMESTAMP}.json" "${BASELINE_DIR}/baseline-latest.json"
    echo "Baseline saved to: $DEST"
    echo "Symlink updated:   ${BASELINE_DIR}/baseline-latest.json"
    exit 0
fi

# --- Determine baseline for comparison ---
if [[ -n "$BASELINE_FILE" ]]; then
    : # use the provided file
elif [[ -f "${BASELINE_DIR}/baseline-latest.json" ]]; then
    BASELINE_FILE="${BASELINE_DIR}/baseline-latest.json"
else
    echo "No baseline found. Run with --save first to create one:" >&2
    echo "  $0 --save" >&2
    exit 1
fi

echo
echo "Comparing against: ${BASELINE_FILE}"
echo "Regression threshold: ${THRESHOLD}%"
echo

python3 "${SCRIPT_DIR}/bench_compare.py" "$BASELINE_FILE" "$MERGED" --threshold "$THRESHOLD"
