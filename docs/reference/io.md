# I/O & Logging (`src/io/`)

> **Maturity:** mature · **Audited:** 2026-05-29 · See [Capability Audit](../CAPABILITY_AUDIT.md) · [Onboarding](../ONBOARDING.md)

## Purpose
The `io` subsystem produces all field output and the logging used across SHOCCS. It writes time-series simulation results as a single XDMF (`.xmf`) XML index file plus one raw little-endian `float64` binary file per dump per variable, laid out so ParaView/VisIt can open the whole run. It also owns the dump-scheduling logic (write every N steps or every Δt of simulation time) and a thin `spdlog` wrapper (`logs`) that is threaded through the entire build tree, not just I/O.

## Where it lives
| File | Role |
| --- | --- |
| `src/io/field_io.hpp` | Public orchestrator class `field_io` + `from_lua` factory — the front door. |
| `src/io/field_io.cpp` | `write()` scheduling/dispatch and `from_lua` Lua-config parsing (the config schema lives here). |
| `src/io/xdmf.hpp` / `xdmf.cpp` | `.xmf` XML generation/append via pugixml; defines the header (`3DCoRectMesh` + `Polyvertex`) and per-dump temporal grid with `Binary` `DataItem` `Seek` offsets. |
| `src/io/field_data.hpp` / `field_data.cpp` | Raw binary payload writer: scalar `D + Rx/Ry/Rz` fields and cut-cell R-point geometry (with the 2D z-swap). |
| `src/io/interval.hpp` | `interval<T>` + `d_interval` dump-scheduling state machines (header-only). |
| `src/io/logging.hpp` / `logging.cpp` | `logs` spdlog wrapper used across the whole project. |
| `src/io/CMakeLists.txt` | Builds `shoccs-logging` and `shoccs-io` libs + the 4 unit tests (note the inconsistent ctest labels). |
| `src/systems/detail/scalar_system_utils.hpp` | `write_scalar_error`: the shared bridge from `heat`/`scalar_wave` into `field_io::write`. |
| `src/simulation/simulation_cycle.cpp` | Owns the `field_io`, calls `sys.write(...)` at step 0 and each accepted step. |

## Public API / entry points

### `field_io` — orchestrator (`field_io.hpp`)
```cpp
field_io() = default;                              // empty/disabled writer

field_io(xdmf&&,
         field_data&&,
         d_interval&&,
         std::string&& io_dir,
         int suffix_length,
         const logs& = {});

bool write(std::span<const std::string> names,
           std::span<const scalar_view> scalars,
           const step_controller& controller,
           real dt,
           std::array<std::span<const mesh_object_info>, 3> R);   // returns true iff a dump happened

static std::optional<field_io> from_lua(const sol::table&, const logs& = {});
```
`names`, `scalars`, and the generated file names are **parallel, equal-length** spans (`names[i]` ↔ `scalars[i]`). `R` is the mesh's cut-cell R-points (`mesh::R()` = `{Rx(), Ry(), Rz()}`); pass `{}` (empty spans) when there is no embedded geometry.

### `xdmf` — XML index writer (`xdmf.hpp`)
```cpp
xdmf() = default;
xdmf(std::string xmf_filename, index_extents ix, domain_extents bounds);

void write(int grid_number,
           real time,
           std::span<const std::string> var_names,
           std::span<const std::string> file_names,
           std::array<std::span<const mesh_object_info>, 3>,
           const logs& logger) const;
```
`grid_number == 0` rewrites the file from a fresh header; later calls reload and append one temporal grid.

### `field_data` — binary payload writer (`field_data.hpp`)
```cpp
field_data() = default;
field_data(index_extents ix);

void write(std::span<const scalar_view> scalars,
           std::span<const std::string> filenames) const;       // D then Rx/Ry/Rz, concatenated

void write_geom(std::span<const std::string> filenames,
                std::array<std::span<const mesh_object_info>, 3>) const;  // rx/ry/rz point coords
```

### `interval<T>` / `d_interval` — dump scheduling (`interval.hpp`)
```cpp
template <typename T> class interval {
    interval();                              // never fires (no distance) — operator bool() == false
    interval(T distance, T first = T{});
    bool operator()(T val, T tolerance = T{}) const;   // val - first >= distance - tolerance
    operator bool() const;                   // true iff a distance is configured
    interval& operator++();                  // advance: first += distance  (also operator-- / postfix)
};

class d_interval {
    d_interval() = default;
    d_interval(interval<int> step_interval, interval<real> time_interval);
    bool operator()(const step_controller& step, real dt);     // should we dump now?
    int  current_dump() const;               // running dump counter (NOT step number)
    d_interval& operator++();                // advance whichever sub-interval(s) fired + bump counter
};
```

### `logs` — spdlog wrapper (`logging.hpp`)
```cpp
struct logs {
    logs() = default;                                                  // disabled
    logs(const std::string& logging_dir, bool enable, const std::string& logger_name);
    logs(bool enable, const std::string& logger_name);                // stdout sink
    logs(const logs& parent, const std::string& logger_name);         // inherit enabled flag, new stdout logger
    logs(const logs& parent, const std::string& logger_name, const std::string& file_name); // file sink

    void set_pattern(const std::string& pat);
    operator bool() const;                                             // enabled?
    void operator()(spdlog::level::level_enum lvl, const std::string& msg) const;
    template <typename... Args>
    void operator()(spdlog::level::level_enum lvl, fmt::format_string<Args...>, Args&&...) const;
};
```

### Lua config (`simulation.io` block)
Parsed in `field_io::from_lua` (`field_io.cpp:69`). All keys optional:

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `write_every_step` | int | (none) | Dump every N accepted steps. |
| `write_every_time` | real | (none) | Dump every Δt of *simulation* time. |
| `dir` | string | `"io"` | Output directory (created on step 0). |
| `suffix_length` | int | `6` | Zero-pad width of the dump counter in file names. |
| `xdmf_filename` | string | `"view.xmf"` | Name of the `.xmf` index file (placed under `dir`). |

If neither `write_every_*` is set, both intervals are disabled and `write()` always returns `false` (no output). Shipped configs that enable I/O: `heat.lua` (`write_every_time = 0.01`, 2D), `scalar_wave.lua` (`write_every_step = 1`), `lua-configs/brady_livescu_4_3*.lua` (`write_every_step = 10`/`100`).

## How it works
Build path: `Lua simulation.io` → `field_io::from_lua` → a `field_io` owning `{xdmf, field_data, d_interval, logs}`. `from_lua` first calls `cartesian::from_lua(tbl)` (so it needs a valid `mesh` block); if that fails it returns `nullopt`. If there is no `io` sub-table it returns a default-constructed, **valid but disabled** `field_io{}` (not `nullopt`).

Per-step flow, driven from `simulation_cycle::run` (`simulation_cycle.cpp:70`, `:108`):

1. `sys.write(io, reg, ref, controller, dt)` → `heat::write` / `scalar_wave::write` (`heat.cpp:404`, `scalar_wave.cpp:399`) build a `{U, Error}` pair and delegate to `detail::write_scalar_error` (`scalar_system_utils.hpp:301`), which calls `io.write(io_names, io_scalars, c, dt, m.R())`.
2. `field_io::write` (`field_io.cpp:32`) asks `dump_interval(step, dt)`; if it returns `false`, bail out returning `false`.
3. On step 0 it `create_directories(io_dir)`.
4. It builds per-variable file names `"<var>.<NNNNNN>"` using the **dump counter** `dump_interval.current_dump()` zero-padded to `suffix_length`.
5. `xdmf_w.write(...)` appends a temporal grid to the `.xmf` (rewriting the file header on grid 0).
6. On the *first* dump (`n == 0`) it writes the cut-cell point geometry to `rx`/`ry`/`rz` via `field_data_w.write_geom`.
7. `field_data_w.write(...)` writes the raw binary payloads (one file per variable).
8. `++dump_interval` advances the fired interval(s) and bumps the dump counter; return `true`.

**Output format.** One human-unreadable `.xmf` XML index (`view.xmf`) references per-dump binary files. Binaries are bare little-endian `float64`, **no header**, with the four components concatenated in order `D, Rx, Ry, Rz`; the XDMF `Seek` attribute (`xdmf.cpp:76`, offset accumulated in `sub_grid` at `:80`) tells the reader where each component starts in the file. The interior field uses a `3DCoRectMesh` topology (`Origin_DxDyDz` geometry from `domain_extents`); cut-cell points use `Polyvertex` topologies (`RX`/`RY`/`RZ` sub-grids) pointing at the shared `rx`/`ry`/`rz` coordinate files. Intended consumer: **ParaView**.

`d_interval::operator()` fires when the step interval is ready **or** the time interval is ready **or** `(int)step == 0` (`interval.hpp:78`). The time branch passes `dt` as the `tolerance` argument to `interval<real>::operator()` so it fires on the step that first reaches the target simulation time under variable `dt`.

## How to extend

**Add a new output format (e.g. HDF5/VTK).** Mirror `xdmf`/`field_data`: a stateful writer class with a `write()` taking `var_names` + `file_names` + the R-geometry array. Add it as a member of `field_io`, call it from `field_io::write` alongside `xdmf_w`/`field_data_w`, register the `.cpp` in `src/io/CMakeLists.txt`'s `add_library(shoccs-io ...)`, and add an `add_unit_test`.

**Add a new config key.** Extend `field_io::from_lua` (`field_io.cpp:69`) to read from the `io` sub-table (use `io["key"].get_or(default)` or `sol::optional<...>`), then thread the value into the `field_io` ctor.

**Add a new dump-scheduling policy.** Add an `interval`-like member to `d_interval`, OR it into `d_interval::operator()`, and advance it in `operator++`. Build the new sub-interval in `from_lua`.

**Make a new PDE system emit fields.** Implement its `write()` (signature in `system.hpp:61`): typically delegate to `detail::write_scalar_error` (`scalar_system_utils.hpp`) with a system-specific `io_names` vector. `heat`/`scalar_wave` use `io_names = {"U", "Error"}` (`heat.hpp:35`, `scalar_wave.hpp:42`). To add variables, extend both `io_names` and the matching `io_scalars` vector built in the system's `write()`. Systems that emit nothing (`inviscid_vortex`, `hyperbolic_eigenvalues`, `empty`) use no-op `write` stubs.

## Gotchas & invariants
- **File names use the DUMP COUNTER, not the step number.** `U.000003` is the 4th dump, not step 3 (`field_io.cpp:43,49`; counter is `d_interval::current_dump`). Easy to misread when correlating dumps with log step numbers.
- **Step 0 is always dumped.** `d_interval::operator()` returns `true` when `(int)step == 0` regardless of the configured interval (`interval.hpp:78`), and the output directory is created only on step 0 (`field_io.cpp:41`). The first frame is special-cased.
- **Cut-cell geometry is written only on the first dump** (`n == 0`, `field_io.cpp:53`). The `rx`/`ry`/`rz` files are static across the run — they are not re-emitted per dump.
- **Binary files are opened *without* `std::ios::binary`** (`field_data.cpp` uses plain `std::ofstream`). Harmless on Linux, but the writes are `reinterpret_cast<const char*>` of raw doubles, so a Windows port would suffer newline-translation corruption. Output is also host-endian (XDMF `Binary` format assumes little-endian).
- **2D dimension swap.** In `write_geom`, when `ix[2] == 1` (a 2D run) the R-point position is reordered to `{z, y, x}` before writing (`field_data.cpp:16-20`). This must stay paired with the XDMF geometry ordering; getting it wrong silently mis-locates cut-cell points in ParaView. The 2D `heat.lua` exercises exactly this branch.
- **Parallel-span assumption with no size check.** `field_io::write` assumes `names`, `scalars`, and the derived file-name list are equal-length; `field_data::write` indexes `scalars[idx]` by filename index without checking (`field_data.cpp:39-41`). A mismatch is UB, not a thrown error.
- **`from_lua` returns an empty-but-valid `field_io{}`** (not `nullopt`) when there is no `io` table (`field_io.cpp:75`); `nullopt` only on `cartesian::from_lua` failure. An empty `field_io::write` always returns `false`.
- **pugixml failure → `std::terminate()`.** On any load/save error `xdmf::write` aborts the entire simulation (`xdmf.cpp:151,160`), not a recoverable error.
- **`logs` copies do NOT share a logger.** Each copy ctor builds a *new* `spdlog::logger`/sink (`logging.cpp:21-27`); repeated copies create many loggers with the same name. The `enable` flag does propagate by copy.

## Maturity & known gaps
**Verdict: mature.** Evidence: `field_io` is wired into every concrete system's `write()` and driven by `simulation_cycle::run()` each step; shipped configs (`heat.lua`, `scalar_wave.lua`, `brady_livescu_4_3*.lua`) enable it; core files (`field_io`/`xdmf`/`field_data`) were actively maintained 2026-03-27. All four unit tests pass, and these PASS results are trustworthy because the I/O tests have no `libkokkoscore` runtime dependency (unlike the 23 Kokkos-linked tests).

The subsystem is fully integrated and runtime-reachable — nothing here is dead. The known gaps are all about **missing verification / incomplete coverage** (all four flags audited to `finish`, none to delete):

- **Cut-cell R-point geometry output** (`write_geom` + XDMF `RX/RY/RZ` sub-grids + `rx`/`ry`/`rz` files) — *partial*. Live on the output path of shipped sims (`heat.lua`, a 2D sphere run, even hits the `ix[2]==1` z-swap branch), but every unit test passes empty R-spans (`T{}`), so the z-swap and the Polyvertex/Seek pairing have zero automated assertions; correctness rests on a manual "load in ParaView" check. See [Cleanup Plan](../CLEANUP_PLAN.md).
- **Output-content correctness** (binary payload + `.xmf`) — *partial*. Tests assert only bool returns / no-crash; nothing reads back the `.xmf` XML (Dimensions/Precision/Format/Seek) or the binary bytes, and `field_data.cpp` has no test file at all. The cross-consistency between `field_data`'s write order and `xdmf`'s `Seek` arithmetic is exactly the regression no current test catches. See [Cleanup Plan](../CLEANUP_PLAN.md).
- **`write_every_time` end-to-end** — *partial*. `interval<real>` is unit-tested in isolation, but the time-driven `d_interval` path with a real adaptive `step_controller` and the `dt`-as-tolerance firing logic is covered by no C++ test (`grep d_interval` across `*.t.cpp` = zero hits); `heat.lua` depends on it. See [Cleanup Plan](../CLEANUP_PLAN.md).
- **ctest label inconsistency** — *partial* (config drift, not a code bug). `t-logging` and `t-field_io` are labeled `"io"`; `t-interval` and `t-xdmf` are labeled `"shoccs-io"` (`CMakeLists.txt:4,14,15,16`). Git history shows this was unintended drift (interval was originally `"io"`, changed to `"shoccs-io"` in Jan 2021, then xdmf copy/pasted the mistake). Consequence: no single label selects exactly the four io tests, and because `ctest -L` uses unanchored regex, `-L io` over-matches `t-simulation_cycle` (its `simulation` label contains "io"). Fix: change `"shoccs-io"` → `"io"` on lines 14-15. See [Cleanup Plan](../CLEANUP_PLAN.md).

## Tests
Four unit tests (`src/io/CMakeLists.txt`), all PASS:
- `t-logging` (`logging.t.cpp`, label `io`) — enable/disable, no output when disabled.
- `t-interval` (`interval.t.cpp`, label `shoccs-io`) — `interval<T>` in isolation: never-fire, no-rollover, rollover firing count. Plain values; no `step_controller`, no `dt`-as-tolerance, no `d_interval`.
- `t-xdmf` (`xdmf.t.cpp`, label `shoccs-io`) — writes header at grid 0, appends grid 1 to a temp `.xmf`. Only `REQUIRE` checks the test's own empty input; the `.xmf` is never read back.
- `t-field_io` (`field_io.t.cpp`, label `io`) — default no-io path returns `false`; full write path with 2 scalars returns `true`. The data test is explicitly comment-marked "one needs to load the output in paraview".

**Not covered:** output file *contents* (byte layout, Seek offsets, endianness, XML structure are never read back/asserted); cut-cell `Rx/Ry/Rz` geometry (all tests pass `T{}`); the 2D z-swap branch; `write_every_time` end-to-end through an adaptive run; `from_lua`'s `xdmf_filename`/`suffix_length`/`dir` overrides. There is no `field_data.t.cpp`. No disabled or commented-out tests in this subsystem. Run with `ctest --test-dir build -R 't-(logging|interval|xdmf|field_io)'` (the `-L` labels are inconsistent — see gaps above).

## Related docs
- [Capability Audit](../CAPABILITY_AUDIT.md), [Onboarding](../ONBOARDING.md), [Cleanup Plan](../CLEANUP_PLAN.md).
- Reference docs for upstream subsystems: `simulation` (owns/drives `field_io`), `systems` (`write` fan-out + `write_scalar_error` bridge), `mesh` (supplies `index_extents`, `domain_extents`, and `R()` cut-cell points), `temporal` (`step_controller`), `fields` (`scalar_view`/`scalar_span`).
- **Legacy (pre-Kokkos rationale archive):** `plans/07-simulation-io-mms.md` describes the original range-v3 implementation (`vs::transform`, `rs::to`, `vs::zip`, `vs::repeat_n`) — all removed during the Kokkos migration; the current code uses explicit loops and `ccs::repeat_n`. Its "Build-breakage status" and 7.x checklist are historical migration notes, not a description of the shipped code.
- `CLAUDE.md:77` summarizes io as "XDMF/binary field output, spdlog-based logging" — accurate but omits the dump-scheduling layer and cut-cell geometry output.
