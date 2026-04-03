# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when
working with code in this repository. It explains how to *modify*
this code. For guidance on how to *use* this library in
applications, see USAGE.md. When Cvxium is installed, agents can
retrieve USAGE.md at runtime:

```python
import cvxium
cvxium.usage()
```

## What is Cvxium?

Cvxium is a framework for building fast Interior Point Method (IPM)
solvers for convex optimization problems. It provides base classes
that handle the IPM loop (barrier method, centering steps,
backtracking line search) while subclasses supply problem-specific
math: objectives, gradients, Hessians, and constraints. The key value
proposition is exploiting special Hessian structure (diagonal,
diagonal-plus-low-rank, arrow sparsity) to achieve linear-time Newton
steps instead of cubic-time, yielding orders-of-magnitude speedups on
large problems.

## Build and Test

```bash
# Build a wheel
uv build --wheel

# Run all tests
uv run python -m pytest

# Run a specific test
uv run python -m pytest test/test_file.py::TestClass.test_method

# Format code
uv run python -m black cvxium/ test/

# Lint
uv run python -m ruff check cvxium/ test/

# Check typing
uv run python -m mypy
```

## Architecture

### Class Hierarchy (optimization.py)

```
Optimizer (ABC)
├── PhaseISolver (ABC) — find a feasible point, no objective
│
└── BaseInteriorPointMethodSolver — IPM loop, centering, BTLS
    ├── InteriorPointMethodSolver — inequality-only optimization
    ├── PhaseIInteriorPointSolver (+ PhaseISolver) — IPM-based feasibility
    ├── EqualityConstrainedInteriorPointMethodSolver — equality + inequality
    │     with custom barrier initialization via SVD of A
    └── UnconstrainedNewtonSolver — no constraints, pure Newton's method
```

To implement a new solver, pick a base class from this table:

| Constraints           | Optimization                                                             | Feasibility                                                              |
|-----------------------|--------------------------------------------------------------------------|--------------------------------------------------------------------------|
| None                  | UnconstrainedNewtonSolver                                                | n/a                                                                      |
| Equality              | EqualityConstrainedNewtonSolver                                          | EqualitySolver                                                           |
| Inequality            | InteriorPointMethodSolver                                                | PhaseIInteriorPointSolver                                                |
| Equality + Inequality | InteriorPointMethodSolver + EqualityConstrainedInteriorPointMethodSolver | PhaseIInteriorPointSolver + EqualityConstrainedInteriorPointMethodSolver |

### Key abstract methods to implement

- `evaluate_objective(x)` — f0(x)
- `gradient(x)` — nabla f0(x)
- `constraints(x)` — vector of fi(x) values
- `grad_constraints(x)` — Jacobian of constraints
- `hessian_multiply(x, y, t)` — nabla^2 ft(x) @ y (exploit structure here)
- `calculate_newton_step(x, t)` — solve KKT system (exploit structure here)
- `evaluate_dual(x, t)` — Lagrangian dual for convergence certification

### Modules

- **optimization.py** — Base solver classes, IPM loop, centering step,
  BTLS, result dataclasses (`OptimizationSettings`, `NewtonResult`,
  `InteriorPointMethodResult`).
- **numerical_helpers.py** — Structured linear system solvers:
  `solve_diagonal`, `solve_rank_one_update`, `solve_rank_p_update`,
  `solve_with_schur`, `solve_block_plus_one`,
  `solve_arrow_sparsity_pattern`, `solve_kkt_system`. These are the
  building blocks for fast Newton steps.
- **linear_programs.py** — Concrete solvers for LPs: `EqualitySolver`
  (A x = b), `EqualityWithBoundsSolver` (A x = b, x >= 0),
  `EqualityWithBoundsAndImbalanceConstraintSolver`.
- **quadratic_programs.py** — Concrete solvers for QPs:
  `QuadraticNewtonSolver` (unconstrained),
  `QuadraticProgramEqualityBoundsSolver` (A x = b, x >= 0).
- **exceptions.py** — Exception hierarchy carrying diagnostic data
  (suboptimality, last iterate, gradient info). Exceptions like
  `CenteringStepError` and `InteriorPointMethodError` include the last
  iterate so callers can recover partial solutions.

### Design patterns

- **Numerical helpers are composable.** A Hessian that is a low-rank
  update to an arrow-sparse matrix can nest `solve_rank_p_update`
  around `solve_arrow_sparsity_pattern`.
- **Phase I solvers chain.** Optimization solvers with constraints
  need a feasible starting point. Phase I solvers find one, and can
  themselves be built with Cvxium. Constraints can be applied in
  layers, chaining Phase I solvers.
- **Caching for expensive constraints.** When `constraints(x)` calls
  an expensive model, cache on `x.data.tobytes()`. Use multi-level
  caching to avoid computing gradients/Hessians during BTLS trial
  evaluations (which only need constraint values).

## Publishing to PyPI

Pushing a tag matching `v*` triggers the GitHub Actions workflow in
`.github/workflows/publish.yml`, which builds the package with `uv
build` and publishes to PyPI via trusted publishing (no API token
needed). To release a new version:

```bash
# Update the version in pyproject.toml, commit, then tag and push
git tag v0.1.0
git push origin v0.1.0
```

The workflow runs in the `pypi` environment and requires `id-token:
write` permissions for OIDC trusted publishing.
