# Cvxium

[![CI](https://github.com/rwilson4/Cvxium/actions/workflows/ci.yml/badge.svg)](https://github.com/rwilson4/Cvxium/actions/workflows/ci.yml)

Cvxium (pronounced "Calcium") is a Python framework for building fast
Interior Point Method (IPM) solvers for convex optimization problems of
the form:

```
minimize    f0(x)
subject to  A x = b
            fi(x) <= 0,  i = 1, ..., n
```

The framework's distinguishing feature is a clean interface for
exploiting **Hessian structure** to accelerate Newton steps. A generic
solver inverts an n×n dense matrix at each iteration — O(n³). By
encoding the Hessian's structure (diagonal, low-rank update, arrow
sparsity pattern, etc.) through a pair of callables, the same iteration
can run in O(n) or O(r²n), where r is the rank of a low-rank component.
At n = 200 and r = 1, this is roughly a 40× speedup over the naïve
approach.

Cvxium is used by [PyRake](https://github.com/rwilson4/PyRake) to solve
survey weight calibration problems, where it provides custom solvers for
KL-divergence, squared-L2, and Huber distance objectives with equality,
imbalance, and norm constraints.

## Installation

```bash
pip install cvxium
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add cvxium
```

## Quick start: ready-made solvers

For the most common problem types, Cvxium ships concrete solvers that
require no subclassing.

### Find x satisfying Ax = b, x ≥ lb

`EqualityWithBoundsSolver` finds a feasible point (Phase I) and, if
requested, minimizes `‖x‖₂²` subject to the constraints (Phase II):

```python
import numpy as np
from cvxium import EqualityWithBoundsSolver, OptimizationSettings

A = np.random.randn(20, 100)   # p=20 equality constraints, M=100 variables
w_true = np.random.rand(100) + 0.1
b = A @ w_true
lb = 0.01

solver = EqualityWithBoundsSolver(A=A, b=b, lb=lb)

# Phase I: find any strictly feasible point
result = solver.solve()
assert np.all(result.solution > lb)
assert np.allclose(A @ result.solution, b)

# Phase II: minimize ‖x‖₂²
result = solver.solve(fully_optimize=True)
```

The solver detects infeasibility via the dual certificate and raises
`ProblemCertifiablyInfeasibleError` when the problem has no solution.

### Find x satisfying Ax = b, x ≥ lb, ‖Bx − c‖∞ ≤ ψ

`EqualityWithBoundsAndImbalanceConstraintSolver` adds an L∞ imbalance
constraint, useful when exact balance on a subset of covariates is
required alongside a bound on a larger set:

```python
from cvxium import EqualityWithBoundsAndImbalanceConstraintSolver

solver = EqualityWithBoundsAndImbalanceConstraintSolver(
    A=A, b=b, lb=lb,
    B=B, c=c, psi=0.05,  # ‖Bx − c‖∞ ≤ 0.05
)
result = solver.solve()
```

### Solve a quadratic program with equality and bound constraints

`QuadraticProgramEqualityBoundsSolver` solves:

```
minimize    x^T Q x + c^T x
subject to  A x = b
            x >= xl
```

It accepts optional `Q_vector_multiply` and `Q_solve` callables to
exploit structure in Q (see below):

```python
from cvxium import QuadraticProgramEqualityBoundsSolver

solver = QuadraticProgramEqualityBoundsSolver(Q=Q, c=c, A=A, b=b, xl=xl)
result = solver.solve()

print(result.solution)        # optimal x
print(result.objective_value) # primal objective
print(result.dual_value)      # dual lower bound (duality gap = objective - dual)
print(result.nits)            # outer IPM iterations
print(result.inner_nits)      # Newton iterations per centering step
```

#### Exploiting Q structure

When Q has structure, pass `Q_vector_multiply` and `Q_solve` callables.
Both must accept a `scale` and `diag_add` keyword argument — this lets
the framework fold the barrier diagonal directly into the structured
solve, giving the maximum speedup:

```python
# Q = diag(q) + u u^T  (diagonal + rank-one)
q = np.array([...])
u = np.array([...])

def Q_vector_multiply(v, scale=1.0, diag_add=None):
    """Compute (scale * Q + diag(diag_add)) @ v."""
    e = diag_add if diag_add is not None else 0.0
    if v.ndim == 1:
        return (scale * q + e) * v + scale * np.dot(u, v) * u
    return (scale * q + e)[:, np.newaxis] * v + scale * np.outer(u, u @ v)

def Q_solve(v, scale=1.0, diag_add=None):
    """Solve (scale * Q + diag(diag_add)) x = v via Sherman-Morrison."""
    e = diag_add if diag_add is not None else 0.0
    eff_diag = scale * q + e
    coeff = scale / (1.0 + scale * np.dot(u, u / eff_diag))
    if v.ndim == 1:
        v_d = v / eff_diag
        return v_d - coeff * np.dot(u, v_d) * (u / eff_diag)
    v_d = v / eff_diag[:, np.newaxis]
    return v_d - coeff * np.outer(u / eff_diag, u @ v_d)

solver = QuadraticProgramEqualityBoundsSolver(
    Q=Q, c=c, A=A, b=b, xl=xl,
    Q_vector_multiply=Q_vector_multiply,
    Q_solve=Q_solve,
)
```

At n = 200, a diagonal + rank-one Q with `scale`+`diag_add` callables
runs roughly 12× faster than the naïve O(n³) path.

## Building a custom solver

Cvxium's real power is its framework for implementing new problem types.
The design follows the Interior Point Method as described in Boyd &
Vandenberghe (2004): an outer loop increases the barrier parameter t
until the solution converges, and an inner Newton loop solves the
centering problem at each t.

### Class hierarchy

```
Optimizer (ABC)
├── InteriorPointMethodSolver
│   └── EqualityConstrainedInteriorPointMethodSolver
│       └── EqualityWithBoundsSolver  (concrete)
└── PhaseISolver
    ├── EqualitySolver  (concrete, SVD-based)
    └── PhaseIInteriorPointSolver
        └── EqualityWithBoundsSolver  (also concrete Phase I)
```

To implement a new solver, subclass
`EqualityConstrainedInteriorPointMethodSolver` (or
`InteriorPointMethodSolver` for problems without equality constraints).

### Required methods

| Method | Purpose |
|--------|---------|
| `calculate_newton_step(x, t)` | Solve the KKT system for the Newton step; the main performance lever |
| `evaluate_objective(x)` | Return f0(x) as a float |
| `hessian_multiply(x, t, y)` | Return H(x, t) @ y |
| `constraints(x)` | Return vector of fi(x) values (must be < 0 strictly inside feasible region) |
| `grad_constraints(x)` | Return matrix of constraint Jacobians |
| `evaluate_dual(lmbda, nu, x_star)` | Evaluate the dual function; critical for Phase I infeasibility detection |

### Optional overrides (for performance)

| Method | Default | Fast implementation |
|--------|---------|---------------------|
| `grad_constraints_multiply(x, y)` | Dense matrix multiply | Exploit sparsity (e.g., `return -y` for bound constraints) |
| `grad_constraints_transpose_multiply(x, y)` | Dense matrix multiply | Same |
| `btls_keep_feasible(x, delta_x)` | Iterative bisection | Closed-form per-component analysis |

### Example: a norm-constrained feasibility solver

PyRake's `EqualityWithBoundsAndNormConstraintSolver` illustrates a
complete custom solver. It finds x satisfying Ax = b, x > lb, and
‖x‖₂² < φ by minimizing ‖x‖₂² with the IPM and stopping as soon as
the bound is satisfied. The Hessian of the objective 2t‖x‖₂² is the
scaled identity — so `_hessian_ft_diagonal(x, t) = 2t + (x − lb)⁻²`
is the full diagonal, and the Newton step reduces to
`solve_kkt_system(..., hessian_solve=solve_diagonal_eta_inverse, ...)`.
When an imbalance constraint B x − c ≤ ψ is present, the Hessian
gains two rank-q updates (one per sign), handled via
`solve_rank_p_update` nested inside `solve_kkt_system`. The total cost
is O(p³ + p²M) rather than O(M³).

## Numerical helpers

`numerical_helpers.py` provides a composable family of structured linear
solvers. Each accepts an `A_solve` callable — allowing arbitrary
nesting — and handles both 1-D (single right-hand side) and 2-D
(multiple right-hand sides) inputs.

| Helper | Structure of H | Cost |
|--------|---------------|------|
| `solve_diagonal(b, eta)` | `diag(eta)` | O(n) |
| `solve_diagonal_eta_inverse(b, eta_inverse)` | `diag(eta)`, η⁻¹ given | O(n) |
| `solve_arrow_sparsity_pattern(b, eta, zeta, theta)` | `diag(eta)` + last row/col | O(n) |
| `solve_rank_one_update(b, kappa, A_solve, **kw)` | `A + κκᵀ` | O(n) via Sherman-Morrison |
| `solve_rank_p_update(b, kappa, A_solve, **kw)` | `A + κκᵀ` (rank-p κ) | O(r²n + r³) via Woodbury |
| `solve_block_plus_one(b, a12, a22, A11_solve, **kw)` | Block `[A11, a12; a12ᵀ, a22]` | Schur complement |
| `solve_with_schur(b, A12, A22, A11_solve, **kw)` | Block `[A11, A12; A12ᵀ, A22]` | Schur complement |
| `solve_kkt_system(A, g, hessian_solve, **kw)` | KKT `[H, Aᵀ; A, 0]` | O(p²n + p³) |

Helpers compose by passing one as the `A_solve` argument of another:

```python
# Diagonal + rank-one update
x = solve_rank_one_update(b, kappa, A_solve=solve_diagonal, eta=eta)

# Arrow + rank-p update
x = solve_rank_p_update(
    b, kappa,
    A_solve=solve_arrow_sparsity_pattern,
    eta=eta, zeta=zeta, theta=theta,
)

# KKT system with a structured Hessian
delta_x, nu = solve_kkt_system(
    A, g,
    hessian_solve=solve_rank_one_update,
    kappa=kappa,
    A_solve=solve_diagonal,
    eta=eta,
)
```

`solve_kkt_system` implements the Schur complement approach (Boyd &
Vandenberghe, Algorithm C.4). It accepts `hessian_solve` plus any
`**kwargs` forwarded to it, making any composed solver usable as a
drop-in Hessian inverter.

## Exception hierarchy

```
BacktrackingLineSearchError
├── ConstraintBoundaryError       — step would violate a constraint
├── InvalidDescentDirectionError  — Newton step is not a descent direction
└── SevereCurvatureError          — backtracking condition never satisfied

OptimizationError
├── CenteringStepError            — inner Newton loop failed
└── InteriorPointMethodError      — outer IPM loop failed

ProblemInfeasibleError            — no feasible point exists
ProblemCertifiablyInfeasibleError — dual certificate proves infeasibility
ProblemMarginallyFeasibleError    — feasible set is non-empty but has no interior
```

## Optimization settings

`OptimizationSettings` controls the IPM:

```python
from cvxium import OptimizationSettings

settings = OptimizationSettings(
    barrier_multiplier=10.0,    # factor by which t increases each outer iteration
    outer_tolerance=1e-8,       # duality gap threshold for convergence
    outer_tolerance_soft=None,  # looser threshold for feasibility-only problems
    max_outer_iterations=100,
    max_inner_iterations=100,
    verbose=False,
)
```

## Development

```bash
uv sync
uv run python -m pytest            # run all tests
uv run python -m black cvxium/ test/
uv run python -m ruff check cvxium/ test/ --fix
uv run python -m mypy
```

## References

- Boyd, Stephen and Vandenberghe, Lieven. *Convex Optimization*.
  Cambridge University Press, 2004.
