# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when
working with code in this repository.

## Subpackage Purpose

`pyrake/optimization` is a self-contained framework for building
efficient Interior Point Method (IPM) solvers for convex optimization
problems of the form:

    minimize    f0(x)
    subject to  A * x = b
                fi(x) <= 0, i=1,...,n

The "special sauce" is exploiting Hessian structure to achieve fast
Newton steps. This package is used by `pyrake/calibration/` but is
designed to be reusable independently.

## Testing

Tests live in `test/optimization/`. Run from the repo root:

```bash
# Run all optimization tests
uv run python -m pytest test/optimization/

# Run a single test
uv run python -m pytest test/optimization/test_numerical_helpers.py::test_solve_diagonal
```

## Class Hierarchy

```
Optimizer (ABC)
├── InteriorPointMethodSolver(Optimizer)
│   └── EqualityConstrainedInteriorPointMethodSolver(InteriorPointMethodSolver)
│       └── EqualityWithBoundsSolver (also inherits PhaseIInteriorPointSolver)
└── PhaseISolver(Optimizer)
    ├── EqualitySolver(PhaseISolver)         # SVD-based, no inequality constraints
    └── PhaseIInteriorPointSolver(InteriorPointMethodSolver, PhaseISolver)
        └── EqualityWithBoundsSolver
```

`EqualityWithBoundsSolver` in `linear_programs.py` is the canonical
example of a concrete solver: it uses `EqualitySolver` as its Phase I
solver, and implements all abstract methods.

## Implementing a New Solver

Subclass `EqualityConstrainedInteriorPointMethodSolver` (or
`InteriorPointMethodSolver` for problems without equality constraints).
The minimum viable implementation requires:

**Required abstract methods:**
- `calculate_newton_step(x, t)` — solve the KKT system for the
  Newton step; most critical for performance
- `evaluate_objective(x)` — return f0(x) as a float
- `hessian_multiply(x, t, y)` — return H(x, t) @ y
- `constraints(x)` — return vector of fi(x) values (must be < 0
  strictly inside feasible region)
- `grad_constraints(x)` — return matrix of constraint gradients
- `evaluate_dual(lmbda, nu, x_star)` — return dual function value;
  critical for Phase I infeasibility detection

**Optional overrides (for performance):**
- `grad_constraints_multiply(x, y)` — compute `grad_constraints @ y`
  without forming the full matrix
- `grad_constraints_transpose_multiply(x, y)` — compute
  `grad_constraints.T @ y`
- `btls_keep_feasible(x, delta_x)` — analytically compute max step
  size keeping x + s*delta_x strictly feasible; the base
  implementation is iterative and slow

For `EqualityConstrainedInteriorPointMethodSolver`, also override:
- `initialize_barrier_parameter(x0)` — Boyd & Vandenberghe §11.3.1
  gives two formulas; take `max(1.0, t1, t2)`
- `augment_previous_solution(phase1_res)` — extend the Phase I
  solution to the Phase II variable layout
- `finalize_solution(x)` — strip augmented variable before returning

## Numerical Helpers (`numerical_helpers.py`)

These solve `H * x = b` for structured H. They compose via the
`A_solve` callable pattern and accept both 1D (single RHS) and 2D
(multiple RHS) `b`.

| Helper                         | Structure                      | Complexity       |
|--------------------------------|--------------------------------|------------------|
| `solve_diagonal`               | `diag(eta)`                    | O(M)             |
| `solve_diagonal_eta_inverse`   | `diag(eta)`, eta⁻¹ given       | O(M)             |
| `solve_arrow_sparsity_pattern` | `diag(eta) + last-row/col`     | O(M)             |
| `solve_rank_one_update`        | `A + kappa*kappa^T`            | O(2t + 6M)       |
| `solve_rank_p_update`          | `A + kappa*kappa^T` (rank-p)   | O((p+q)t + 2Mp²) |
| `solve_block_plus_one`         | block `[A11, a12; a12^T, a22]` | Schur complement |
| `solve_with_schur`             | block `[A11, A12; A12^T, A22]` | Schur complement |
| `solve_kkt_system`             | KKT `[H, A^T; A, 0]`           | O(p³ + p²M)      |

**Composition pattern** — nest helpers by passing one as `A_solve` to
another. Example from the tests:

```python
# Diagonal + rank-one update:
x = solve_rank_one_update(b, kappa, A_solve=solve_diagonal, eta=eta)

# Arrow sparsity + rank-p update:
x = solve_rank_p_update(b, kappa, A_solve=solve_arrow_sparsity_pattern,
                        eta=eta, zeta=zeta, theta=theta)

# Full KKT with arrow Hessian:
delta_x, nu = solve_kkt_system(A, g,
                               hessian_solve=solve_arrow_sparsity_pattern,
                               eta=eta, zeta=zeta, theta=theta)
```

`solve_kkt_system` takes `hessian_solve` plus any `**kwargs` forwarded
to it. This lets you pass a composed solver as a single callable.

## Exception Hierarchy

```
BacktrackingLineSearchError
├── ConstraintBoundaryError       — even tiny steps violate constraints
├── InvalidDescentDirectionError  — Newton step isn't a descent direction
└── SevereCurvatureError          — backtracking condition never met

OptimizationError
├── CenteringStepError            — inner Newton loop failed
└── InteriorPointMethodError      — outer IPM loop failed
```

`ProblemInfeasibleError` and `ProblemCertifiablyInfeasibleError` (in
`optimization.py`) are raised when feasibility checks fail.

## Key Algorithmic Invariants

- All inequality constraints `fi(x) <= 0` must be **strictly satisfied**
  (`< 0`) at every iterate; `btls_keep_feasible` enforces this.
- The Phase I objective is a slack variable `s`; `is_feasible` returns
  True when `s < 0` at the inner problem's solution.
- `EqualityConstrainedInteriorPointMethodSolver` uses a custom barrier
  parameter initialization (Boyd & Vandenberghe §11.3.1) that typically
  reduces outer iterations compared to a fixed starting value.
- When `A` is rank-deficient, `solve_kkt_system` falls back from
  Cholesky to SVD automatically.
- `EqualitySolver.svd_A` is cached (`@cache`) and uses a QR-then-SVD
  trick when `A` has many more columns than rows (>1000 cols, aspect
  ratio > 10:1), saving time by working with the smaller `R` factor.
