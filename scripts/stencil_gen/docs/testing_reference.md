# stencil_gen Testing Reference

## 1. Test Organization

### File-to-Module Mapping

| Test File | Source Module(s) | Focus |
|---|---|---|
| `test_interior.py` (180 lines, 29 tests) | `stencil_gen.interior` | Interior stencil coefficients: `StencilSpec`, `derive_interior`, gamma/delta arrays |
| `test_boundary.py` (833 lines, 41 tests) | `stencil_gen.taylor_system`, `stencil_gen.boundary`, `stencil_gen.conservation` | Boundary row derivation (E4u/E6u/E8u), conservation constraints |
| `test_temo.py` (1855 lines, 127 tests) | `stencil_gen.temo` | TEMO cut-cell extension: dimensions, uniform boundary, degenerate stencil, psi field, Vandermonde, Neumann, E2_1/E2_2 integration |
| `test_e4_cut_cell.py` (2717 lines, 131 tests) | `stencil_gen.temo`, `stencil_gen.codegen`, `stencil_gen.conservation` | E4_1 cut-cell pipeline: uniform boundary, zeros, conservation, TEMO, code generation, Mathematica workflow |
| `test_phs.py` (1627 lines, 82 tests) | `stencil_gen.phs` | PHS+poly stencils, RBF/tension spline weights, diff matrices, conservation penalty, modified wavenumber, stability |
| `test_group_velocity.py` (1714 lines, 52 tests) | `stencil_gen.group_velocity` | Group velocity analysis: core computation, boundary/interior/cut-cell, 2D, psi sweeps, ray tracing |
| `test_codegen.py` (567 lines, 29 tests) | `stencil_gen.codegen`, `stencil_gen.printer` | C++ code generation: CSE, formatting, struct generation, test file generation |
| `test_printer.py` (72 lines, 10 tests) | `stencil_gen.printer` | Custom C++ code printer: `Pow`, `Rational`, `Integer`, symbol mapping |
| `test_codegen_e4u.py` (229 lines, 5 tests) | `stencil_gen.interior`, `stencil_gen.codegen` | End-to-end E4u_1 pipeline: derive + generate + compile C++ |
| `test_eval_e2_1.py` (60 lines, 3 tests) | `tools/eval_e2_1` | E2_1.cpp fixture evaluator regression tests |

**Total: 509 tests across 10 files (~9,950 lines of test code).**

### Naming Conventions

- **Test files:** `test_<topic>.py` -- topic names match source modules or feature areas.
- **Test functions:** `test_<what>` for standalone functions; `test_<scheme>_<aspect>` for scheme-specific (e.g., `test_E4u_row0_symbolic`, `test_e2_1_dimensions`).
- **Test classes:** `Test<Component>` (e.g., `TestDimensions`, `TestPHSCore`, `TestE4CodeGeneration`). Classes group related tests that share fixtures.
- **Boundary test naming:** follows step numbering from the derivation plan (20.3a--20.3g, 21.1b, etc.).

---

## 2. Running Tests

All commands assume you are in `/workspace/scripts/stencil_gen/`.

### Default Fast Suite

```bash
SYMPY_CACHE_SIZE=50000 uv run pytest tests/ -x -q
```

Runs 509 tests, skipping 105 slow-marked tests. The `-x` flag stops on first failure. `SYMPY_CACHE_SIZE=50000` is essential for performance.

### Including Slow Tests

```bash
SYMPY_CACHE_SIZE=50000 uv run pytest tests/ -x -q --run-slow
```

Runs all 509 tests without skipping any. Slow tests add significant time (minutes to tens of minutes for the full E4 pipeline tests).

### Single File

```bash
SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_interior.py -x -q
SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_temo.py -x -q
```

### Single Class

```bash
SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_temo.py::TestDimensions -x -q
SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_e4_cut_cell.py::TestE4UniformBoundary -x -q
```

### Single Test

```bash
SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_interior.py::test_spec_order -x -q
SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_temo.py::TestDimensions::test_e2_1_dimensions -x -q
```

### By Keyword

```bash
# Skip the heaviest slow tests
SYMPY_CACHE_SIZE=50000 uv run pytest tests/ -x -q -k "not TestMathematicaWorkflow and not TestPolynomialFullStencil and not TestE4CodeGeneration"

# Run only conservation-related tests
SYMPY_CACHE_SIZE=50000 uv run pytest tests/ -x -q -k "conservation"

# Run only E2-related tests
SYMPY_CACHE_SIZE=50000 uv run pytest tests/ -x -q -k "e2"

# Run only slow tests
SYMPY_CACHE_SIZE=50000 uv run pytest tests/ -x -q --run-slow -k "slow"
```

### With Timing Info

```bash
# Show durations for the 20 slowest tests
SYMPY_CACHE_SIZE=50000 uv run pytest tests/ -x -q --durations=20

# Show durations including setup/teardown (reveals fixture costs)
SYMPY_CACHE_SIZE=50000 uv run pytest tests/ -x -q --durations=20 --durations-min=0.5
```

### Collecting Without Running (Dry Run)

```bash
SYMPY_CACHE_SIZE=50000 uv run pytest tests/ --co -q           # list all tests
SYMPY_CACHE_SIZE=50000 uv run pytest tests/ --co -q --run-slow -k "slow"  # list slow tests only
```

---

## 3. Fixture Architecture

### conftest.py: Shared Fixtures

Located at `tests/conftest.py`. Provides the following:

#### Custom CLI Option

- **`--run-slow`** -- Added via `pytest_addoption`. The `pytest_collection_modifyitems` hook skips any test marked `@pytest.mark.slow` unless this flag is passed.

#### Session-Scoped Fixtures

| Fixture | Scope | Returns | Purpose |
|---|---|---|---|
| `assert_taylor_accuracy` | session | function `_check_taylor_accuracy(B_u, q, nu)` | Asserts each row of a boundary matrix satisfies Taylor moment conditions up to the required order. Used across multiple test files. |

#### Module-Scoped Pipeline Caches

These fixtures run expensive symbolic derivations once per module and share the result across all tests in that module:

| Fixture | Scope | Calls | Returns |
|---|---|---|---|
| `e2_1_uniform` | module | `derive_e2_uniform_boundary(nu=1)` | `UniformResult` for E2, first derivative |
| `e2_2_uniform` | module | `derive_e2_uniform_boundary(nu=2)` | `UniformResult` for E2, second derivative |
| `e4u_pipeline` | module | `run_pipeline(p=2)` | `(updated_rows, solution_dict, w_syms, result)` for E4u |
| `e6u_pipeline` | module | `run_pipeline(p=3)` | Same tuple for E6u |
| `e8u_pipeline` | module | `run_pipeline(p=4)` | Same tuple for E8u |

The `run_pipeline` helper chains `derive_boundary` + `build_conservation_system` + `solve_conservation`.

### Per-File Fixtures

#### test_e4_cut_cell.py

| Fixture | Scope | Purpose |
|---|---|---|
| `e4_1_cut_cell_scheme` | module | Caches `derive_cut_cell_scheme(E4_1, psi)` -- shared by `TestDeriveCutCellScheme` and `TestE4CutCellSchemeWithZeros` |
| `e4_1_cut_cell_scheme_conserve` | module | Caches `derive_cut_cell_scheme(E4_1, psi, conserve=True)` |
| `e4_1_cut_cell_conservation` | module | Builds E4_1 cut-cell conservation system for infeasibility tests |
| `mathematica_result` | module | Caches `derive_cut_cell_mathematica` result |
| `e4_result` | class (TestE4UniformBoundary) | `derive_uniform_boundary_for_temo(E4_1)` |
| `e4_zeroed` | class | `derive_uniform_boundary_for_temo(E4_1, zeros={3, 4})` |
| `e4_zeroed_cut_cell` | class | Zero-constrained cut-cell stencil |
| `conservation_solution` | class | Zero-constrained conservation solve |
| `e4_temo` | class | Full E4_1 TEMO pipeline |
| `e4_spec` | class | `StencilGenSpec` built from conserve=True derivation |
| `e4_code` | class | Generated C++ code string |
| `conserved` | class | `derive_uniform_boundary_for_temo(E4_1, conserve=True)` |
| `conserved_cut_cell` | class | Cut-cell stencil with uniform conservation |
| `temo_output` | class | Raw TEMO output before conservation |
| `poly_rows` | class | Polynomial boundary rows for E4_1 |
| `fraction_free_solution` | class | Fraction-free conservation solve |
| `poly_stencil` | class | Stencil with polynomial boundary rows |
| `full_result` | class | Full `derive_cut_cell_scheme(E4_1, psi)` result |

#### test_temo.py

| Fixture | Scope | Purpose |
|---|---|---|
| `e2_2_neumann` | instance (TestNeumannStencil) | E2_2 Neumann stencil derivation |
| `e2_1` | instance (TestE2_1Integration) | Full E2_1 cut-cell stencil |
| `e2_2` | instance (TestE2_2Integration) | Full E2_2 cut-cell stencil with all BC variants |
| `e2_1_scheme` | class (TestE2_1DeriveCutCellSchemeRegression) | `derive_cut_cell_scheme(E2_1, psi)` |
| `e2_1_manual` | class | Manual E2_1 pipeline for regression comparison |

#### test_group_velocity.py

| Fixture | Scope | Purpose |
|---|---|---|
| `e4_classical` | class (TestBoundaryClassical) | E4 boundary rows with conservation and known-good alpha |
| `e2_1_cut_cell` | class (TestCutCellGroupVelocity) | E2_1 cut-cell stencil (symbolic in psi and alpha) |

#### test_codegen_e4u.py

| Fixture | Scope | Purpose |
|---|---|---|
| `e4u_data` | module | Transforms `e4u_pipeline` conftest fixture into codegen-ready data |

#### test_eval_e2_1.py

| Fixture | Scope | Purpose |
|---|---|---|
| `eval_fn` | module | Builds E2_1 evaluator from C++ source file |

---

## 4. Slow Test System

### Mechanism

Tests are marked with `@pytest.mark.slow`. The `conftest.py` hook `pytest_collection_modifyitems` adds a skip marker to all slow-marked tests unless `--run-slow` is passed on the command line. The marker is declared in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow (deselected by default, use --run-slow to include)",
]
```

### Slow Test Inventory (105 tests)

| File | Slow Tests | Slow Classes/Functions |
|---|---|---|
| `test_e4_cut_cell.py` | 101 | `TestE4CutCellConservationSolution`, `TestE4TEMOConstruction`, `TestE4CodeGeneration`, `TestE4TestFileGeneration`, `TestDeriveCutCellScheme`, `TestE4CutCellSchemeWithZeros`, `TestE4UniformConservation`, `TestCutCellConservationAfterUniform`, `TestPolynomialStructure`, `TestPolynomialBoundaryRows`, `TestFractionFreeConservation`, `TestApproachAInfeasibility`, `TestPolynomialFullStencil`, `TestMathematicaWorkflow`, plus standalone functions `test_e4_1_conservation_with_zeros`, `test_e4_1_conservative_taylor_accuracy`, `test_e4_1_conservative_psi_interior`, `test_e4_1_conservation_constant_weights_infeasible_r5`, `test_e4_1_psi_dependent_conservation_infeasible` |
| `test_phs.py` | 3 | `TestModifiedWavenumber::test_e2_boundary_at_optimal_sigma`, `test_e4_boundary_at_optimal_sigma`, `test_dispersion_comparison` |
| `test_group_velocity.py` | 1 | `TestPsiSweepGroupVelocity::test_e4_1_psi_sweep` |

### Why Tests Are Marked Slow

- **E4_1 cut-cell derivation** involves large symbolic matrices (5x7 with rational functions of psi) requiring expensive SymPy simplification, Groebner basis computation, and polynomial factoring.
- **Code generation tests** run the full pipeline from derivation through C++ output and file comparison.
- **Mathematica workflow tests** reproduce the reference Mathematica notebook computation end-to-end.
- **Modified wavenumber/dispersion tests** in test_phs.py run numerical eigenvalue sweeps over parameter ranges.
- **Psi sweep tests** in test_group_velocity.py evaluate stencils at many psi values for stability analysis.

### xfail Tests

One test uses `@pytest.mark.xfail`:

- `test_e4_1_conservation_fails_without_zeros` -- Documents that E4_1 cut-cell conservation is structurally infeasible without zero constraints (alpha_3=alpha_4=0). Proven via Groebner basis.

---

## 5. Test Tiers

### Fast Tier (default, ~404 tests)

Runs in seconds to low minutes. No `--run-slow` needed.

| File | Fast Tests | Typical Content |
|---|---|---|
| `test_interior.py` | 29 | `StencilSpec` properties, known coefficient values, polynomial exactness, input validation |
| `test_boundary.py` | 41 | Taylor system shape/entries, single-row solver, E4u/E6u/E8u conservation and polynomial exactness |
| `test_temo.py` | 127 | Dimensions, uniform boundary, degenerate stencil, psi field operations, Vandermonde, Neumann, E2_1/E2_2 integration |
| `test_e4_cut_cell.py` | 30 | `TestE4UniformBoundary`, `TestE4UniformBoundaryWithZeros`, `TestE4ZeroConstrainedCutCell`, `TestBuildCutCellConservationSystem`, `TestMathematicaUniformConservation`, `test_e4_1_conservation_fails_without_zeros` (xfail) |
| `test_phs.py` | 79 | PHS core, interior weights, E2/E4 boundary comparison, diff matrices (RBF + tension), eigenvalues, Gaussian RBF, tension spline, conservation penalty, stability infrastructure |
| `test_group_velocity.py` | 51 | Core group velocity, interior/boundary/cut-cell analysis, psi sweeps (E2_1 only), 2D, GKS diagnostic, varying coefficients |
| `test_codegen.py` | 29 | CSE, formatting, interior/NBS method generation, struct generation, test file generation |
| `test_printer.py` | 10 | Pow, Rational, Integer, symbol map printing |
| `test_codegen_e4u.py` | 5 | E4u_1 end-to-end: interior coefficients, floating/Dirichlet numerical, full pipeline, compile check |
| `test_eval_e2_1.py` | 3 | E2_1.cpp evaluator regression: specific psi/alpha values, edge cases |

### Slow Tier (105 tests, requires `--run-slow`)

| File | Slow Tests | Why Slow |
|---|---|---|
| `test_e4_cut_cell.py` | 101 | Full E4_1 cut-cell pipeline: TEMO construction, conservation solve, polynomial structure analysis, Groebner basis infeasibility proofs, C++ code generation, Mathematica workflow reproduction |
| `test_phs.py` | 3 | Numerical eigenvalue sweeps for boundary modified wavenumber at optimal tension parameters |
| `test_group_velocity.py` | 1 | E4_1 psi sweep across [0,1] with strict bounds checking |

---

## 6. Coverage Map

### Source Modules and Their Test Coverage

| Source Module | Primary Test File(s) | Aspects Covered |
|---|---|---|
| `stencil_gen.interior` | `test_interior.py`, `test_codegen_e4u.py` | `StencilSpec` properties, `derive_interior`, gamma/delta arrays, polynomial exactness, input validation |
| `stencil_gen.taylor_system` | `test_boundary.py` | `build_taylor_system` shape, entries, RHS |
| `stencil_gen.boundary` | `test_boundary.py` | `solve_boundary_row`, `BoundaryRow` |
| `stencil_gen.conservation` | `test_boundary.py`, `test_e4_cut_cell.py` | `build_conservation_system`, `solve_conservation`, `_interior_contribution` |
| `stencil_gen.temo` | `test_temo.py`, `test_e4_cut_cell.py` | All TEMO functions: dimensions, uniform boundary, degenerate stencil, psi field, Vandermonde, solve_temo_row, cut-cell stencil construction, Neumann, assembly, `derive_cut_cell_scheme`, Mathematica workflow |
| `stencil_gen.codegen` | `test_codegen.py`, `test_codegen_e4u.py`, `test_e4_cut_cell.py` | CSE, formatting, struct generation, test file generation, `StencilGenSpec`, `compute_test_values` |
| `stencil_gen.printer` | `test_printer.py`, `test_codegen.py` | `StencilCodePrinter`, `build_symbol_map` |
| `stencil_gen.phs` | `test_phs.py` | PHS weights, RBF/tension boundaries, diff matrices, conservation penalty, modified wavenumber, stability |
| `stencil_gen.group_velocity` | `test_group_velocity.py` | All group velocity functions: core, boundary, cut-cell, 2D, psi sweep, ray trace, GKS diagnostic, nonuniform, varying coefficient |
| `stencil_gen._util` | (indirect) | Utility functions tested indirectly through higher-level modules |
| `tools/eval_e2_1` | `test_eval_e2_1.py` | E2_1.cpp fixture evaluator |

### Modules Without Dedicated Tests

- `stencil_gen.__init__` -- Package init, no logic.
- `stencil_gen.__main__` -- CLI entry point, not directly tested.
- `sweeps/*` -- Sweep scripts (alpha extraction, epsilon sweep, tension sweep, etc.) are not covered by the pytest suite; they are standalone analysis scripts.

---

## 7. Adding Tests

### Where to Put New Tests

- **New tests for an existing module:** Add to the corresponding test file. For example, new `stencil_gen.phs` tests go in `test_phs.py`.
- **New module:** Create `test_<module_name>.py` in `tests/`.
- **Cross-module integration tests:** Place in the test file for the highest-level module involved (e.g., `test_e4_cut_cell.py` for E4 pipeline integration, `test_temo.py` for E2 pipeline integration).

### How to Use Fixtures

1. **Reuse conftest fixtures** for E2 uniform boundary results (`e2_1_uniform`, `e2_2_uniform`) and pipeline results (`e4u_pipeline`, `e6u_pipeline`, `e8u_pipeline`). These are module-scoped and avoid redundant derivation.

2. **Use `assert_taylor_accuracy`** to validate boundary matrices:
   ```python
   def test_my_stencil_taylor(self, assert_taylor_accuracy, e2_1_uniform):
       assert_taylor_accuracy(e2_1_uniform.B_u, q=1, nu=1)
   ```

3. **Create class-scoped fixtures** when multiple tests share an expensive intermediate result:
   ```python
   class TestMyFeature:
       @pytest.fixture(scope="class")
       def my_result(self):
           return expensive_derivation()

       def test_shape(self, my_result):
           assert my_result.shape == (3, 4)

       def test_accuracy(self, my_result):
           # reuses same my_result instance
           ...
   ```

4. **Use `pytest.approx`** for numerical comparisons rather than element-wise assertion loops:
   ```python
   assert values == pytest.approx(expected, abs=1e-12)
   ```

### When to Mark Slow

Mark a test `@pytest.mark.slow` when it:

- Runs the full E4_1 (or higher order) cut-cell derivation pipeline.
- Performs Groebner basis or other expensive symbolic algebra.
- Generates C++ code from a full derivation.
- Runs numerical sweeps over large parameter ranges.
- Takes more than ~5 seconds in isolation.

Apply the marker to the class if all tests in the class are slow:
```python
@pytest.mark.slow
class TestMyExpensivePipeline:
    ...
```

Or to individual tests:
```python
@pytest.mark.slow
def test_my_expensive_computation(self):
    ...
```

### Essential Environment Variable

Always set `SYMPY_CACHE_SIZE=50000` when running tests. The default SymPy cache size of 1000 causes severe performance degradation with the large symbolic expressions in this codebase.

### Test Class Structure Pattern

The codebase follows a consistent pattern for test classes:

1. A class-scoped or instance-scoped fixture computes the expensive result once.
2. Individual test methods validate specific aspects: shape, free symbols, Taylor accuracy, conservation, known values, limit behavior (psi=0 degenerate, psi=1 uniform).
3. Tests are named to indicate what property they check: `test_<scheme>_<property>` (e.g., `test_e2_1_shape`, `test_e2_1_taylor_accuracy`, `test_e2_1_uniform_limit`).
