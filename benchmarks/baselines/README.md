# Benchmark Baselines

Stored baseline results for regression tracking.

## Usage

Save a new baseline after building with `-DBUILD_BENCHMARKS=ON`:
```bash
./scripts/bench_compare.sh --save
```

Compare current performance against the latest baseline:
```bash
./scripts/bench_compare.sh
```

Compare against a specific baseline:
```bash
./scripts/bench_compare.sh benchmarks/baselines/baseline-20260320-024030.json
```

## Files

- `baseline-latest.json` — symlink to the most recent baseline.
- `baseline-<timestamp>.json` — timestamped snapshots.

## Notes

Baselines are machine-specific. Results vary with hardware, OS, compiler,
and Kokkos backend. When switching environments, create a fresh baseline
with `--save` before using comparison mode.

The default regression threshold is 10%. Override via:
```bash
BENCH_THRESHOLD=5 ./scripts/bench_compare.sh
```
