# Utilities (`src/utils/`)

> **Maturity:** mature · **Audited:** 2026-05-29 · See [Capability Audit](../CAPABILITY_AUDIT.md) · [Onboarding](../ONBOARDING.md)

## Purpose
`src/utils/` is a minimal, header-only utility directory. Today it holds exactly one type: `ccs::bounded<T>`, a numeric value that carries a half-open `[min, max)` range and reports — via `operator bool()` — whether it is still inside that interval. It exists to model monotonic counters/accumulators whose "am I done?" check is the value going out of range. Despite the generic name, this is no longer a grab-bag: `extents.hpp/.cpp` lived here historically but were deleted as dead code in the 2026-03-27 Phase 18 cleanup, leaving `bounded` as the sole resident.

## Where it lives
| File | Role |
| --- | --- |
| `src/utils/bounded.hpp` | The entire public API: the `bounded<T>` class template. Header-only. |
| `src/utils/bounded.t.cpp` | Catch2 unit test (`t-bounded`, label `utils`). |
| `src/utils/CMakeLists.txt` | Defines the `shoccs-utils` INTERFACE library and registers the test. |

## Public API / entry points

All in `namespace ccs`, from `src/utils/bounded.hpp`:

```cpp
template <Numeric T>          // Numeric = std::integral<T> || std::floating_point<T>  (src/types.hpp:24)
struct bounded {
    // --- construction ---
    bounded() = default;                  // value=min=max=0  -> empty range [0,0), always false
    bounded(T t_max);                     // value=0, min=0, max=t_max
    bounded(T t, T t_min, T t_max);       // explicit value, min, max

    // --- queries / conversions ---
    operator bool() const;                // true iff (t_min <= t) && (t < t_max)   [half-open]
    operator T()    const;                // implicit conversion back to the held value

    // --- mutation (changes the value only; bounds are fixed at construction) ---
    bounded& operator+=(T dt);
    bounded& operator++();    bounded operator++(int);
    bounded& operator--();    bounded operator--(int);
};
```

Notes on the constructors:
- `bounded(T t_max)` is the common one — it makes a counter that starts at `0` with lower bound `0` and the supplied upper bound. This is how `step_controller` builds both of its counters from Lua.
- The three private members `t`, `t_min`, `t_max` have no accessors other than the two conversion operators; only `t` is mutable (through the increment/compound-add operators).

## How it works

`bounded<T>` is a value plus an immutable half-open interval. The only "logic" is `operator bool()`, which is `(t_min <= t) && (t < t_max)`. Everything else just moves `t`.

The single production consumer is the time integrator's `step_controller` (`src/temporal/step_controller.hpp`):

- Members `bounded<int> step` and `bounded<real> time` hold the step count and accumulated simulation time (lines 14-15).
- `step_controller::from_lua` (`src/temporal/step_controller.cpp`) reads `max_step` and `max_time` from the Lua `step_controller` table and constructs `bounded<int>{max_step}` and `bounded<real>{max_time}` — each a counter starting at `0`.
- `step_controller::advance(dt)` does `time += dt; step += 1;`, walking both counters forward.
- `step_controller::operator bool()` (line 35) is exactly `time && step` — i.e. "both counters are still in range." `simulation_cycle`'s time-stepping loop continues while this is true, so `bounded` directly drives loop termination.

Data flow: Lua `step_controller{max_step, max_time}` → `step_controller::from_lua` → two `bounded` counters → `advance()` each step → `operator bool()` (`time && step`) → simulation loop exit.

## How to extend

**To add a new small utility:** follow the established header-only pattern.
1. Create `src/utils/<name>.hpp` (header-only; the `shoccs-utils` target is INTERFACE and lists no `.cpp` sources). Put it in `namespace ccs`.
2. Add `src/utils/<name>.t.cpp` and register it in `src/utils/CMakeLists.txt`:
   `add_unit_test(<name> "utils" shoccs-utils)` — the helper is defined at `/workspace/CMakeLists.txt:51` and links `Catch2::Catch2WithMain` plus the libraries you pass.
3. Other subsystems can `#include "utils/<name>.hpp"` directly; they do **not** need to link `shoccs-utils` because every subsystem already adds `src/` to its own include path (see Gotchas).

**To extend `bounded` itself:** keep the `Numeric` constraint, and respect that `t_min`/`t_max` are fixed at construction by design — add new accessors/operators rather than mutating the bounds, or the half-open `[min, max)` contract that `step_controller` relies on will change semantics.

## Gotchas & invariants

- **Default-constructed `bounded<T>{}` is permanently `false`.** It has `value = min = max = 0`, i.e. the empty range `[0, 0)`, so `operator bool()` is always `false` no matter how you `++` or `+=`. This is intentional and relied upon: a default `step_controller` (no Lua config) runs zero steps, which is the desired behavior for eigenvalue-only analysis runs (`step_controller.cpp` forces `max_step = 0` when neither `max_step` nor `max_time` is specified). The `"bounded default"` test asserts this.
- **The interval is half-open `[t_min, t_max)`.** `t == t_max` is out of range (`false`); `t == t_min` is in range (`true`). Easy off-by-one trap if you assume inclusive bounds.
- **Bounds are immutable.** `t_min`/`t_max` are private with no setters; `+=`, `++`, `--` move only the value. There is no way to rescale the range after construction.
- **Two implicit conversions coexist** (`operator bool` and `operator T`). For `bounded<int>` this can produce surprising implicit-conversion ambiguity in arithmetic/boolean contexts. The one real consumer dodges this by exposing explicit named accessors on `step_controller` (`operator int()`, `operator real()`, `simulation_step()`, `simulation_time()`).
- **No `#pragma once` / include guard.** `bounded.hpp` opens directly with `#include "types.hpp"`. It is currently pulled in through exactly one header chain, so this has never bitten, but a second includer in the same translation unit could trigger a redefinition.
- **Not self-contained for arbitrary include roots.** `bounded.hpp` includes `"types.hpp"` with a bare relative path, relying on `src/` being on the include path.
- **`shoccs-utils` is a vestigial-but-exported CMake target.** It is an INTERFACE library, but its only consumer of the header (`temporal/shoccs-integrate`, which includes `utils/bounded.hpp`) does **not** link it — temporal resolves the include through its own `target_include_directories(... ${SRC}/..)`. `shoccs-utils` is referenced only by its own `t-bounded` test and the top-level `install()`/`EXPORT shoccs` list (`CMakeLists.txt:85`). See "Maturity & known gaps."

## Maturity & known gaps

**Verdict: mature.** Tiny, but real and exercised. Evidence:
- Real production caller: `step_controller` (`src/temporal/`) uses `bounded<int> step` and `bounded<real> time` as data members, and `step_controller::operator bool()` (`time && step`) is the simulation-loop termination check. `bounded` directly drives time-stepping exit.
- Dedicated test `t-bounded` (label `utils`) passes.
- The implementation is complete for its scope: all increment/decrement/compound-add operators and both conversion operators are present.
- Stable: `bounded.hpp` last changed in 2021 (commit `b8d341f`, "filled out step_controller") and survived the Phase 18 dead-code cleanup untouched, while sibling `extents.hpp/.cpp` were removed in that same pass.

**Partial item (verified flag):**
- **`shoccs-utils` INTERFACE target — its include-resolution role is vestigial** (`src/utils/CMakeLists.txt`). Confirmed: a whole-repo grep for `shoccs-utils` returns exactly three hits — the library definition, the `t-bounded` test link, and the top-level `install(... shoccs-utils ... EXPORT shoccs)` list. No production subsystem links it; `temporal` resolves `utils/bounded.hpp` via its own `src/..` include path. The target **is** built, tested, and intentionally part of the installed `shoccs::` package export, so it is **not dead** — verdict is **keep**, not delete. Removing it would silently drop `utils/bounded.hpp` from the exported interface. If cleanup is ever desired, the safe action is to document it as an exported header-only convenience target and optionally make `temporal` link it explicitly. See [Cleanup Plan](../CLEANUP_PLAN.md).

There are no dead (zero-caller) or experimental items in this subsystem.

## Tests

- **`t-bounded`** (`src/utils/bounded.t.cpp`, label `utils`) — links `Catch2WithMain` via `add_unit_test`, no Kokkos `main()`. Two `TEST_CASE`s:
  - `"bounded default"` — the default-constructed `bounded<int>` (empty `[0,0)` range, always `false`, even after `++` and `+=`).
  - `"bounded"` — `bounded<real>{0,0,4.3}` crossing its upper bound via `+=`, and the single-arg `bounded<real>{4.5}` constructor (min=0) crossing via `++`.
- Run it: `ctest --test-dir build -R t-bounded` or by label `ctest --test-dir build -L utils`.
- **Not covered:** the lower-bound transition (`t < t_min` making it `false` from below) is never exercised — only upper-bound crossings are tested. `operator T()` and `operator--` are not directly asserted. No tests for integral overflow or NaN-in-range edge cases. Coverage is adequate for the documented monotonic-counter use but does not pin the full `[min, max)` semantics. No disabled tests.

## Related docs
- [Temporal reference](temporal.md) — `step_controller`, the only production consumer of `bounded` (and where the `utils ← temporal` dependency materializes; note CLAUDE.md's subsystem list does not currently mention `utils` at all).
- [Core types reference](core-types.md) — source of the `Numeric` concept and the `real` alias that `bounded<T>` is parameterized on.
- [Capability Audit](../CAPABILITY_AUDIT.md) · [Cleanup Plan](../CLEANUP_PLAN.md) · [Onboarding](../ONBOARDING.md)
