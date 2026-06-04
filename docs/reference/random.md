# Random (`src/random/`)

> **Maturity:** mature · **Audited:** 2026-05-29 · See [Capability Audit](../CAPABILITY_AUDIT.md) · [Onboarding](../ONBOARDING.md)

## Purpose
`random` is a tiny convenience wrapper around the C++ standard `<random>` facilities, implementing the well-known "Walter Brown" RNG idiom: one process-global `std::default_random_engine` plus `pick()` helpers that draw uniform ints/reals without forcing every call site to construct its own engine and distribution. **It is test-support infrastructure only** — it exists to generate random input vectors and coefficients for fuzzing the matrix, operator, and mesh code, and is **not** part of the solver/production data path or any Lua config flow.

## Where it lives
| File | Role |
| --- | --- |
| `src/random/random.hpp` | Public API declarations (`global_urng`, `randomize`, two `pick` overloads, `pick_r`) in namespace `ccs`; includes `types.hpp` for the `real` alias. |
| `src/random/random.cpp` | Definitions: function-local-static engine and distributions; `pick()` implementations; `pick_r()` (a dead forwarder — see Maturity). |
| `src/random/CMakeLists.txt` | Defines the `shoccs-random` static library and its PUBLIC include dir via `BUILD_INTERFACE`/`INSTALL_INTERFACE` generator expressions. |

## Public API / entry points
All symbols are free functions in namespace `ccs` (declared in `random.hpp`):

```cpp
// Process-global engine, default-seeded (deterministic) unless randomize() is called.
std::default_random_engine& global_urng();

// Reseed the global engine from std::random_device (non-deterministic from here on).
void randomize();

// Uniform integer in [from, thru] — INCLUSIVE on both ends.
int pick(int from, int thru);

// Uniform real in [from, upto) — default args give [0, 1). Called as pick() in tests.
real pick(real from = 0, real upto = 1);

// DEAD: thin forwarder to pick(real, real), zero callers. See Maturity & known gaps.
real pick_r(real from = 0, real upto = 1);
```

`real` is the project's `double` alias from `src/types.hpp`.

## How it works
The implementation is the standard Walter Brown global-URBG pattern (`src/random/random.cpp`):

- `global_urng()` returns a reference to a **single** function-local `static std::default_random_engine`. Because it is default-constructed, the engine starts from a **fixed seed**, so all draws are reproducible across runs unless reseeded.
- `randomize()` pulls one value from a `std::random_device` and feeds it to `global_urng().seed(...)`, switching all subsequent draws to a non-deterministic stream.
- Each `pick()` overload owns a function-local `static` distribution (`std::uniform_int_distribution<>` / `std::uniform_real_distribution<real>`) and invokes it as `d(global_urng(), param_t{from, upto})`. Passing a fresh `param_type` per call is the trick that lets a **single static distribution serve arbitrary ranges** — the distribution object is reused, but the bounds come from the call.

Flow at a call site (test code): `#include "random/random.hpp"` → `pick(...)` → `global_urng()` (shared engine) → uniform draw. There is no per-call allocation and no engine construction after the first call.

## How to extend
To add a new random helper:

1. Declare a free function in namespace `ccs` in `src/random/random.hpp` and define it in `src/random/random.cpp`.
2. Follow the existing pattern: a function-local `static` distribution plus `d(global_urng(), param_t{...})` so the distribution is reused but parameterized per call. **Always route through `global_urng()`** rather than constructing a new engine, so seeding/determinism stays centralized.

To use randomness from a test target:

1. Add `shoccs-random` to that target's link libraries. For `add_unit_test()` targets, append it as a trailing `ARGN` token (it forwards to `target_link_libraries`); see `src/matrices/CMakeLists.txt:13` (`add_unit_test(csr "matrices" shoccs-matrices shoccs-random)`). For hand-rolled targets, add it to the explicit `target_link_libraries(...)` list (e.g. `src/operators/CMakeLists.txt:24`, `src/mesh/CMakeLists.txt:10`).
2. `#include "random/random.hpp"` in the test source.
3. The engine is default-seeded, so tests are deterministic by default. Call `randomize()` in the test body to opt into non-deterministic fuzzing.

Do **not** introduce this utility into production `systems/` or `temporal/` paths — it is intentionally test-only.

## Gotchas & invariants
- **Determinism trap:** `global_urng()` is default-seeded, so tests reproduce by default — but several tests call `randomize()` (`mesh.t.cpp:268`, `derivative.t.cpp:220/273/398/432`, `circulant.t.cpp:35`, `inner_block.t.cpp:37`), which seeds from `random_device`. A failure in those tests may not reproduce on rerun. Check for `randomize()` before assuming a flaky failure is a logic bug.
- **Integer range is inclusive on both ends:** `pick(lo, hi)` returns a value in `[lo, hi]`. Easy to off-by-one against half-open expectations.
- **`pick()` with no args is a real, not an int:** the no-argument call resolves to the `pick(real, real)` overload and returns a `real` in `[0, 1)`. To get a random integer you MUST pass two `int` args.
- **Not thread-safe:** a single shared global engine plus function-local-static distributions. Concurrent calls (e.g. inside a Kokkos parallel region) would race. Keep usage on the host serial path only.
- **Test scaffolding despite living under `src/`:** it sits alongside production subsystems and is in the install target, but it is purely test support. Do not mistake its `mature` status for production-readiness in numerical code.

## Maturity & known gaps
**Verdict: mature** — meaning a stable, finished test-support utility, not production solver code. Evidence: the implementation is complete, correct, and idiomatic (the standard Walter Brown global-URBG pattern, ~36 lines of `.cpp`); the RNG logic has been stable since 2021 (commit `a952451`), with only CMake hygiene touched recently (commit `36bf718`, plan 15.11a/b). It is actively used as a fixture by the matrices, operators, and mesh test suites. It has no production callers and no tests of its own.

Dead / spurious items in this subsystem (verified):

- **`pick_r(real, real)` — DEAD (zero callers, safe to delete).** Declared at `random.hpp:17`, defined at `random.cpp:32`; a trivial one-line forwarder to the actively-used `pick(real, real)` overload. Repo-wide grep finds only its own declaration and definition. Introduced once in 2021 (`a952451`) and never touched since — a leftover overload-disambiguation attempt. Safe to remove both lines; see [Cleanup Plan](../CLEANUP_PLAN.md).
- **`shoccs-random` linked into `t-shapes` but unused — DEAD link edge.** `src/mesh/CMakeLists.txt:14` (`add_unit_test(shapes "mesh" shoccs-mesh shoccs-random)`) links the library, but `shapes.t.cpp` contains no `#include` of `random/random.hpp` and no `pick`/`randomize` call. A 2021 copy-paste leftover from the (legitimate) mesh-test line. Trim the token to `add_unit_test(shapes "mesh" shoccs-mesh)`; the library itself stays. Note: this makes the common "random is used by the shapes test" framing **incorrect**. See [Cleanup Plan](../CLEANUP_PLAN.md).
- **Commented-out `divergence` test referencing a bare `random` token — DEAD.** `src/operators/CMakeLists.txt:20` (`#add_unit_test(divergence "operators" operators random)`). The `divergence.*` sources were deleted in the Phase 11-18 cleanups, and the bare `random` token predates the current `shoccs-random` naming, so even uncommented it would not resolve. Safe to delete the stale comment line; see [Cleanup Plan](../CLEANUP_PLAN.md).

## Tests
There is **no** test targeting `random` itself — there is no `random.t.cpp`, and `shoccs-random` is never the unit-under-test. The correctness of `pick`/`randomize` is unverified by any assertion. Instead it is a test **dependency**, linked into and `#include`d by:

- **matrices** (label `matrices`): `t-csr`, `t-dense`, `t-block`, `t-circulant`, `t-inner_block` — random input vectors/coefficients.
- **operators** (label `operators`): `t-derivative`, `t-laplacian` — random field values via `pick()`.
- **mesh** (label `mesh`): `t-mesh` — `pick()`/`randomize()` for mesh fuzzing.

Not covered: the RNG functions themselves (no assertions), and `pick_r` (exercised by nothing). No disabled tests. The build is green (fixed 2026-06-04, was the Kokkos 5.1 `create_graph` break) and ctest is 47/48 with one remaining failure (`t-laplacian`); `t-csr` and `t-E2_1` have been fixed. `random` itself has no Kokkos dependency.

## Related docs
- [matrices](matrices.md), [operators](operators.md), [mesh](mesh.md) — the consumers of `pick()`/`randomize()` in their test suites.
- [core-types](core-types.md) — the `real` alias (`src/types.hpp`) used in the API.
- Project [CLEANUP_PLAN](../CLEANUP_PLAN.md) — tracks the three dead items above.
- Note: `CLAUDE.md`'s "Key Subsystems" list (items 1-10) does not mention `random`; this is the canonical reference for it.
