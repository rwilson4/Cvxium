# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

It explains how to *modify* this code. For guidance on how to *use* this library in
applications, see USAGE.md. When Cvxium is installed, agents can retrieve USAGE.md at
runtime:

```python
import cvxium
cvxium.usage()
```

## What is Cvxium?

Cvxium is a framework for fast interior-point solvers for convex optimization. Speed
comes from exploiting structure in the Newton-step Hessian (diagonal, banded, low rank,
arrow, sparse equality constraints) so a Newton step costs linear rather than cubic
time. The `cvxium/__init__.py` docstring is the overview.

## Setup

After cloning, install the pre-commit hook (black, ruff, mypy, pytest):

```bash
cp scripts/pre-commit .git/hooks/pre-commit
```

## Build and test

```bash
uv sync                                   # create .venv with dev dependencies
uv run python -m pytest                   # all tests
uv run python -m pytest test/solvers/test_newton_solver.py::NewtonSolverTest
uv run python -m black cvxium/ test/      # format
uv run python -m ruff check cvxium test   # lint
uv run python -m mypy                     # type check (cvxium and test)
uv build --wheel
```

Tests are `unittest.TestCase` classes (parameterized with `parameterized`) run under
pytest; parameterized cases get `_0`, `_1`, ... suffixes. Every test directory has an
`__init__.py`, so module names are `test.solvers.test_hsd_solver` and so on; the perf
baseline is keyed on those names.

Inside the package, modules import each other relatively (`from ..linalg.systems import
...`); tests import from `cvxium.<subpackage>.<module>`. The public API is re-exported
from `cvxium/__init__.py` and its `__all__`; update both when adding a public class.
Each subpackage `__init__.py` holds only a docstring. `cvxium._typing.none_throws`
narrows optionals in expressions where an `assert` would not fit.

### Performance guard

`cvxium/diagnostics/perf_guard.py` fails a test whose solver iteration counts move
materially, up *or* down, against `test/perf_baseline.json`. `test/conftest.py` arms it
by setting `CVXIUM_PERF_BASELINE` before cvxium is imported; set the variable to an empty
string to disable it. If a change intentionally alters iteration counts, regenerate with
`CVXIUM_PERF_BASELINE_WRITE=1 uv run python -m pytest` and explain why in the commit
message. Counts below 8 iterations are not guarded, and known-bimodal tests are listed
under the baseline's `__ignore__` key.

## Architecture

Cvxium separates the *statement* of a problem from the *algorithm* that solves it.

- **`problems/`**: the statement. `Problem` (`problem.py`) is built fluently,
  `Problem.minimize(obj).subject_to(eq=..., ineq=[...])`, from an `Objective` (or `None`
  for a feasibility problem), a list of inequality `ConstraintLayer`s, and an optional
  `EqualityConstraintLayer` (A w = b). It owns two pieces of shared assembly:
  `build_hessian_system` (sums the objective's and each layer's weighted Hessian
  contributions into a structured `MatrixSystem`) and `make_phase1` (builds the Phase I
  problem that finds a strictly feasible start). Concrete layers live in
  `linear_constraints.py` and `quadratic_constraints.py`; every layer subclasses either
  `SlackConstraintLayer` or `DirectConstraintLayer`, and that choice determines how
  `make_phase1` handles it. `cone.py` holds symmetric cones and Nesterov-Todd scalings
  for the HSD solver. `result.py` defines `SolveResult`, a frozen composition of parts
  (spine, `objective`, `duals`, solver-specific `diagnostics`, optional `certificate`
  and `phase1`) rather than an inheritance chain; it imports no other Cvxium module, to
  avoid import cycles.
- **`solvers/`**: the algorithms, all subclasses of `Solver` (`solver.py`):
  `BarrierSolver`, `PrimalDualSolver`, `NewtonSolver`, `HsdSolver`. `Solver` also
  carries the shared Phase I cascade (`_solve_feasibility` builds
  `Problem.make_phase1` and solves it with the same solver). `EqualitySolver` projects
  onto A x = b and supplies the Schur-complement barrier initialization.
  `optimization.py` holds the older subclassing-based hierarchy
  (`InteriorPointMethodSolver`, `FeasibilityInteriorPointSolver`, ...) that runs the
  barrier loop; `BarrierSolver` bridges a `Problem` onto it through the private
  `_BarrierAdapter`. `phase1.py` holds the lifting layers for slack-form Phase I.
  `settings.py` groups all knobs into `OptimizationSettings` (outer, inner, Armijo,
  line search, primal-dual, observability, `trace`); it depends only on
  `diagnostics.trace`, so layers can import it without pulling in the solver engine.
- **`linalg/`**: structured linear algebra. `numerical_helpers.py` has the raw
  `solve_*`/`multiply_*` kernels; `systems.py` arranges them into a composable tree of
  `MatrixSystem`s (`DiagonalSystem`, `BandedSystem`, `LowRankUpdatedSystem`,
  `ArrowSystem`, `BlockDiagonalSystem`, `PermutedSystem`, ...) that `plus` together
  while preserving structure. The indefinite KKT matrix is a `SaddleSystem`, not a
  `MatrixSystem`; an injected `SaddleFactory` on the equality layer selects the variant
  (Schur complement default, sparse, banded, arrow).
- **`saddle/`**: two-block stationarity (saddle-point) problems, max over x, min over
  y, with their own barrier (`StationaritySolver`) and primal-dual
  (`PrimalDualStationaritySolver`) engines and `TwoBlockSolveResult`.
- **`frontier/`**: `MultiObjectiveOptimizer` sweeps a trade-off parameter to trace a
  Pareto frontier; `frontier_interpolation.py` interpolates between traced points.
- **`diagnostics/`**: `Trace` receives structured solver events (`on_*` hooks);
  `LoggingTrace` (default) writes to `cvxium.*` loggers, `ListTrace`/`DataFrameTrace`
  collect in memory (pandas is an optional dependency, imported lazily). The `cvxium`
  logger is silent (WARNING) by default; users opt in via `ObservabilitySettings(verbose=True)`
  or `enable_logging`.
- **`exceptions.py`**: exception hierarchy carrying diagnostic data (last iterate,
  suboptimality) so callers can recover partial solutions.

When adding a solver capability, the usual extension points are a new
`ConstraintLayer` (values, Jacobian products, barrier, `hessian_contribution`,
step-length bound) or a new `MatrixSystem`/`SaddleSystem` for a Hessian structure; the
solvers consume these through `Problem` without changes.

## Typing notes

mypy checks both `cvxium` and `test`. NumPy's stubs type `np.linalg.solve`, `inv` and
`lstsq` results as `floating[Any]` and `np.bincount` as integer, so code that needs
`NDArray[np.float64]` wraps them in `np.asarray(..., dtype=np.float64)` or
`.astype(np.float64, copy=False)`. `x.shape[k]` is `Any`, which makes
`rng.standard_normal(x.shape[k])` resolve to the scalar overload; pass
`int(x.shape[k])`.

## Publishing to PyPI

Pushing a tag matching `v*` triggers `.github/workflows/publish.yml`, which builds the
package with `uv build` and publishes to PyPI via trusted publishing (no API token
needed). To release, update the version in `pyproject.toml`, commit, then:

```bash
git tag v0.1.0
git push origin v0.1.0
```
